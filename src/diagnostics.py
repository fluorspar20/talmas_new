"""
Diagnostic data collection and visualization for TALMAS denoising analysis.

Captures per-step data during low_confidence_remasking_sample and writes
four files to output_dir/:
  attention.png    — grid of attention heatmaps (last layer, every 10 steps)
  suppression.png  — grid of suppression bias heatmaps (last layer, every 10 steps)
  confidence.png   — per-position confidence at every 10 steps
  scalar.png       — mean attention flowing to [MASK] tokens over all steps

Each attention/suppression heatmap uses 4-color quadrant coding:
  Blue    — real → real      (baseline attention between revealed tokens)
  Red     — real → [MASK]    (what TALMAS suppresses)
  Green   — [MASK] → real
  Magenta — [MASK] → [MASK]  (partially suppressed via μ)

Install DiagnosticsCollector AFTER TALMASHookManager so it wraps the
already-patched block forward.  Remove it BEFORE TALMASHookManager.remove().
"""

import functools
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.config import TALMASConfig
from src.talmas import compute_lambda


class DiagnosticsCollector:

    def __init__(
        self,
        model,
        talmas_cfg: Optional[TALMASConfig],
        num_layers: int,
        capture_attn_every: int = 10,
        capture_conf_every: int = 10,
    ):
        self.talmas_cfg = talmas_cfg
        self.num_layers = num_layers
        self.capture_attn_every = capture_attn_every
        self.capture_conf_every = capture_conf_every

        # Per-step storage
        self._attn: Dict[int, np.ndarray] = {}   # step → (S, S) mean-over-heads
        self._supp: Dict[int, np.ndarray] = {}   # step → (S, S) bias magnitude
        self._conf: Dict[int, np.ndarray] = {}   # step → (L,)  post-clamp confidence
        self._mask: Dict[int, np.ndarray] = {}   # step → (S,)  bool
        self._t_vals: Dict[int, float] = {}       # step → r_t

        self._should_capture = False
        self._latest_attn: Optional[np.ndarray] = None
        self._last_block = None
        self._orig_fwd = None

        self._install_capture(model)

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_capture(self, model) -> None:
        last = None
        for name, module in model.named_modules():
            if "transformer.blocks." in name and name.count(".") == 3:
                last = module
        if last is None:
            print("DiagnosticsCollector: last block not found — attention capture disabled")
            return

        self._last_block = last
        self._orig_fwd = last.forward
        collector = self

        @functools.wraps(self._orig_fwd)
        def capturing_fwd(x, attention_bias=None, **kwargs):
            if not collector._should_capture:
                return collector._orig_fwd(x, attention_bias=attention_bias, **kwargs)

            captured: List[np.ndarray] = []
            old_sdpa = F.scaled_dot_product_attention

            def capturing_sdpa(query, key, value, attn_mask=None, **kw):
                # Call real F.sdpa so model output is unchanged
                out = old_sdpa(query, key, value, attn_mask=attn_mask, **kw)
                # Recompute attention weights separately for visualization only
                scale = query.shape[-1] ** -0.5
                with torch.no_grad():
                    logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
                    if attn_mask is not None:
                        logits = logits + attn_mask.float()
                    weights = torch.softmax(logits, dim=-1)      # (B, H, S, S)
                    captured.append(
                        weights.mean(dim=1)[0].cpu().numpy().astype(np.float32)  # (S, S)
                    )
                return out

            F.scaled_dot_product_attention = capturing_sdpa
            try:
                result = collector._orig_fwd(x, attention_bias=attention_bias, **kwargs)
            finally:
                F.scaled_dot_product_attention = old_sdpa

            if captured:
                collector._latest_attn = captured[0]
            return result

        last.forward = capturing_fwd

    # ------------------------------------------------------------------
    # Sampling loop interface
    # ------------------------------------------------------------------

    def begin_step(self, step_idx: int, t_val: float, mask_positions: torch.Tensor) -> None:
        """Call BEFORE the model forward pass. Arms attention capture and computes suppression."""
        self._t_vals[step_idx] = t_val
        self._latest_attn = None

        m = mask_positions[0].cpu().numpy()   # (S,) bool
        self._mask[step_idx] = m

        capture = (step_idx % self.capture_attn_every == 0)
        self._should_capture = capture

        # Compute suppression bias analytically for the last layer at this step
        if capture and self.talmas_cfg is not None and self.talmas_cfg.lambda_max > 0.0:
            lam = compute_lambda(
                self.talmas_cfg.lambda_max, t_val,
                self.num_layers - 1, self.num_layers,
                use_timestep_gate=self.talmas_cfg.use_timestep_gate,
                use_layer_gate=self.talmas_cfg.use_layer_gate,
                sigmoid_slope=self.talmas_cfg.sigmoid_slope,
                timestep_exponent=self.talmas_cfg.timestep_exponent,
            )
            mf = m.astype(np.float32)
            query_gate = (1.0 - mf[:, None]) + self.talmas_cfg.mu * mf[:, None]  # (S, 1)
            self._supp[step_idx] = (lam * mf[None, :] * query_gate).astype(np.float32)

    def end_step(self, step_idx: int, confidence: torch.Tensor) -> None:
        """Call AFTER confidence is finalized (post zero-EOS and already-unmasked clamping)."""
        self._should_capture = False
        if self._latest_attn is not None:
            self._attn[step_idx] = self._latest_attn
        if step_idx % self.capture_conf_every == 0:
            self._conf[step_idx] = confidence.detach().cpu().float().numpy()

    def remove(self) -> None:
        if self._last_block is not None and self._orig_fwd is not None:
            self._last_block.forward = self._orig_fwd

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self, prompt_len: int, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nGenerating diagnostics in {output_dir}/")

        if self._attn:
            _make_heatmap_grid(
                self._attn, self._mask, self._t_vals, prompt_len,
                title="Attention weights — last layer, mean over heads",
                out_path=os.path.join(output_dir, "attention.png"),
            )
            _plot_scalar(
                self._attn, self._mask, self._t_vals, prompt_len,
                out_path=os.path.join(output_dir, "scalar.png"),
            )

        if self._supp:
            _make_heatmap_grid(
                self._supp, self._mask, self._t_vals, prompt_len,
                title="TALMAS suppression bias — last layer",
                out_path=os.path.join(output_dir, "suppression.png"),
            )

        if self._conf:
            _plot_confidence(
                self._conf, self._t_vals,
                out_path=os.path.join(output_dir, "confidence.png"),
            )


# ---------------------------------------------------------------------------
# Private plot helpers
# ---------------------------------------------------------------------------

def _resp(arr: np.ndarray, p: int) -> np.ndarray:
    """Crop (S, S) matrix to the response×response region."""
    return arr[p:, p:]


def _render_quadrant_heatmap(arr: np.ndarray, mask_resp: np.ndarray, vmax: float) -> np.ndarray:
    """
    Return (L, L, 3) float32 RGB image with 4-color quadrant coding.

    Intensity encodes attention weight (0 = white, max = saturated color).
    Color encodes the interaction type:
      Blue    — real query    → real key      (baseline)
      Red     — real query    → [MASK] key    (suppressed by TALMAS)
      Green   — [MASK] query  → real key
      Magenta — [MASK] query  → [MASK] key   (partially suppressed via μ)
    """
    v = np.clip(arr / (vmax or 1.0), 0, 1).astype(np.float32)  # (L, L)

    m_q = mask_resp[:, None].astype(bool)   # (L, 1) — query is [MASK]
    m_k = mask_resp[None, :].astype(bool)   # (1, L) — key   is [MASK]

    rgb = np.ones((arr.shape[0], arr.shape[1], 3), dtype=np.float32)  # start white

    # real → real  :  subtract v from R and G  →  white→blue
    rr = (~m_q) & (~m_k)
    rgb[:, :, 0] -= np.where(rr, v, 0)
    rgb[:, :, 1] -= np.where(rr, v, 0)

    # real → [MASK]:  subtract v from G and B  →  white→red
    rm = (~m_q) & m_k
    rgb[:, :, 1] -= np.where(rm, v, 0)
    rgb[:, :, 2] -= np.where(rm, v, 0)

    # [MASK] → real:  subtract v from R and B  →  white→green
    mr = m_q & (~m_k)
    rgb[:, :, 0] -= np.where(mr, v, 0)
    rgb[:, :, 2] -= np.where(mr, v, 0)

    # [MASK] → [MASK]:  subtract v from G      →  white→magenta
    mm = m_q & m_k
    rgb[:, :, 1] -= np.where(mm, v, 0)

    return np.clip(rgb, 0, 1)


def _make_heatmap_grid(
    data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    t_vals: Dict[int, float],
    prompt_len: int,
    title: str,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    steps = sorted(data.keys())
    resp_frames = [_resp(data[s], prompt_len) for s in steps]
    vmax = max(f.max() for f in resp_frames) or 1.0
    L = resp_frames[0].shape[0]

    n = len(steps)
    n_cols = min(5, n)
    n_rows = math.ceil(n / n_cols)
    tick_step = max(32, L // 4)
    ticks = list(range(0, L, tick_step))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 3.5 * n_rows + 0.6),
        squeeze=False,
    )

    for idx, step in enumerate(steps):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        mask = mask_data[step][prompt_len:]   # (L,) bool
        rgb = _render_quadrant_heatmap(resp_frames[idx], mask, vmax)

        ax.imshow(rgb, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=5)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=5)
        ax.set_xlabel("Key pos", fontsize=5)
        ax.set_ylabel("Query pos", fontsize=5)
        ax.set_title(
            f"step {step}  r_t={t_vals[step]:.2f}\n{int(mask.sum())}/{L} masked",
            fontsize=6,
        )

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Colour legend
    legend_patches = [
        mpatches.Patch(color="blue",    label="real → real"),
        mpatches.Patch(color="red",     label="real → [MASK]  ← suppressed"),
        mpatches.Patch(color="green",   label="[MASK] → real"),
        mpatches.Patch(color="magenta", label="[MASK] → [MASK]  ← μ-suppressed"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=4,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.0),
        framealpha=0.9,
    )

    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {os.path.basename(out_path)}: {n} panels ({n_rows}×{n_cols} grid)")


def _plot_confidence(
    conf_data: Dict[int, np.ndarray],
    t_vals: Dict[int, float],
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    steps = sorted(conf_data.keys())
    L = len(conf_data[steps[0]])
    xs = np.arange(L)

    norm = Normalize(vmin=steps[0], vmax=steps[-1])
    cmap = cm.plasma

    fig, ax = plt.subplots(figsize=(13, 4))
    for step in steps:
        color = cmap(norm(step))
        ax.plot(xs, conf_data[step], color=color, linewidth=0.8, alpha=0.85)

    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlim(0, L - 1)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Response position")
    ax.set_ylabel("Confidence")
    ax.set_title(
        f"Token confidence per position — every {steps[1] - steps[0] if len(steps) > 1 else '?'} steps"
        "  (light = early, dark = late)"
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Denoising step", fraction=0.02, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  confidence.png saved  ({len(steps)} lines)")


def _plot_scalar(
    attn_data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    t_vals: Dict[int, float],
    prompt_len: int,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    steps = sorted(attn_data.keys())
    scalars: List[float] = []
    r_ts: List[float] = []

    for step in steps:
        attn = attn_data[step]
        mask = mask_data[step]
        attn_resp = attn[prompt_len:]     # (L, S) — response queries
        mask_resp = mask[prompt_len:]     # (L,) bool
        real_resp = ~mask_resp
        if real_resp.any() and mask.any():
            val = float(attn_resp[real_resp, :][:, mask].mean())
        else:
            val = 0.0
        scalars.append(val)
        r_ts.append(t_vals[step])

    t_map = dict(zip(steps, r_ts))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, scalars, "o-", color="#E87B4C", linewidth=1.5, markersize=3)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(
        "Mean attention: real response tokens → [MASK] tokens  (last layer, sampled steps)\n"
        "With suppression this should be lower than baseline at the same step"
    )
    ax.set_ylim(bottom=0)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    sampled = steps[:: max(1, len(steps) // 6)]
    ax2.set_xticks(sampled)
    ax2.set_xticklabels([f"{t_map[s]:.2f}" for s in sampled], fontsize=7)
    ax2.set_xlabel("r_t (mask ratio)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  scalar.png saved")
