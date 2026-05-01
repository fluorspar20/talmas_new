"""
Diagnostic data collection and visualization for TALMAS denoising analysis.

Captures per-step data during low_confidence_remasking_sample and writes
ten files to output_dir/:

  Existing:
    attention.png     — grid of attention heatmaps (last layer, every N steps)
    suppression.png   — grid of suppression bias heatmaps [TALMAS only]
    confidence.png    — per-position confidence traces
    scalar.png        — mean real→[MASK] attention over captured steps

  New (process analysis):
    flow.png          — mean attention weight per quadrant over all steps
    entropy.png       — attention entropy for real vs [MASK] query rows
    trajectory.png    — unmasking trajectory heatmap + first-revelation order
    conf_dist.png     — violin distribution of masked-token confidence per step
    flip_rate.png     — fraction of masked positions that changed prediction
    revelation_conf.png — model confidence at the moment each token is committed

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

        # Sparse storage — captured every N steps
        self._attn: Dict[int, np.ndarray] = {}   # step → (S, S) mean-over-heads
        self._supp: Dict[int, np.ndarray] = {}   # step → (S, S) bias magnitude
        self._conf: Dict[int, np.ndarray] = {}   # step → (L,)  post-clamp confidence

        # Dense storage — captured every step (cheap: bool/float/int arrays)
        self._mask: Dict[int, np.ndarray] = {}      # step → (S,)  bool
        self._t_vals: Dict[int, float] = {}          # step → r_t
        self._conf_all: Dict[int, np.ndarray] = {}  # step → (L,)  confidence
        self._pred_all: Dict[int, np.ndarray] = {}  # step → (L,)  argmax token ids

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

    def end_step(
        self,
        step_idx: int,
        confidence: torch.Tensor,
        pred_ids: torch.Tensor,
    ) -> None:
        """Call AFTER confidence and pred_ids are finalised (post zero-EOS and already-unmasked clamping)."""
        self._should_capture = False
        if self._latest_attn is not None:
            self._attn[step_idx] = self._latest_attn
        if step_idx % self.capture_conf_every == 0:
            self._conf[step_idx] = confidence.detach().cpu().float().numpy()

        # Always store for dense-step plots (cheap)
        self._conf_all[step_idx] = confidence.detach().cpu().float().numpy()
        self._pred_all[step_idx] = pred_ids.detach().cpu().numpy().astype(np.int32)

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
            _plot_attention_flow(
                self._attn, self._mask, self._t_vals, prompt_len,
                out_path=os.path.join(output_dir, "flow.png"),
            )
            _plot_entropy(
                self._attn, self._mask, prompt_len,
                out_path=os.path.join(output_dir, "entropy.png"),
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
            _plot_confidence_distribution(
                self._conf, self._mask, prompt_len,
                out_path=os.path.join(output_dir, "conf_dist.png"),
            )

        if self._mask:
            _plot_trajectory(
                self._mask, prompt_len,
                out_path=os.path.join(output_dir, "trajectory.png"),
            )

        if self._pred_all:
            _plot_flip_rate(
                self._pred_all, self._mask, prompt_len,
                out_path=os.path.join(output_dir, "flip_rate.png"),
            )

        if self._conf_all:
            _plot_revelation_confidence(
                self._conf_all, self._mask, prompt_len,
                out_path=os.path.join(output_dir, "revelation_conf.png"),
            )


# ---------------------------------------------------------------------------
# Private helpers shared across plot functions
# ---------------------------------------------------------------------------

def _resp(arr: np.ndarray, p: int) -> np.ndarray:
    """Crop (S, S) matrix to the response×response region."""
    return arr[p:, p:]


def _quadrant_means(ar: np.ndarray, mask_r: np.ndarray):
    """
    Return (rr, rm, mr, mm) mean attention weights for the 4 interaction types.
    ar       — (L, L) response-region attention
    mask_r   — (L,) bool, True where token is [MASK]
    """
    real_idx = np.where(~mask_r)[0]
    mask_idx = np.where(mask_r)[0]

    def qmean(ri, ki):
        if len(ri) == 0 or len(ki) == 0:
            return 0.0
        return float(ar[np.ix_(ri, ki)].mean())

    return (
        qmean(real_idx, real_idx),
        qmean(real_idx, mask_idx),
        qmean(mask_idx, real_idx),
        qmean(mask_idx, mask_idx),
    )


# ---------------------------------------------------------------------------
# Existing plot helpers (unchanged)
# ---------------------------------------------------------------------------

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

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

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
    print("  scalar.png saved")


# ---------------------------------------------------------------------------
# New plot helpers
# ---------------------------------------------------------------------------

def _plot_attention_flow(
    attn_data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    t_vals: Dict[int, float],
    prompt_len: int,
    out_path: str,
) -> None:
    """
    Mean attention weight per quadrant over captured steps.

    Shows how attention redistributes as denoising progresses: with TALMAS,
    real→[MASK] and [MASK]→[MASK] drop while real→real and [MASK]→real rise
    (softmax mass conservation).
    """
    import matplotlib.pyplot as plt

    steps = sorted(attn_data.keys())
    rr_vals, rm_vals, mr_vals, mm_vals = [], [], [], []

    for step in steps:
        ar = attn_data[step][prompt_len:, prompt_len:]  # (L, L) response region
        mask_r = mask_data[step][prompt_len:]
        rr, rm, mr, mm = _quadrant_means(ar, mask_r)
        rr_vals.append(rr); rm_vals.append(rm)
        mr_vals.append(mr); mm_vals.append(mm)

    fig, ax = plt.subplots(figsize=(10, 4))
    kw = dict(linewidth=1.5, markersize=4)
    ax.plot(steps, rr_vals, "o-", color="steelblue",  label="real  → real",   **kw)
    ax.plot(steps, mr_vals, "^-", color="seagreen",   label="[MASK]→ real",   **kw)
    ax.plot(steps, rm_vals, "s-", color="crimson",    label="real  → [MASK]", **kw)
    ax.plot(steps, mm_vals, "D-", color="darkorchid", label="[MASK]→ [MASK]", **kw)

    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Mean attention weight (response region)")
    ax.set_title(
        "Mean attention weight per interaction type — last layer, response tokens only\n"
        "With TALMAS: real→[MASK] and [MASK]→[MASK] should be suppressed; real→real rises"
    )
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=0)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    sampled = steps[:: max(1, len(steps) // 6)]
    ax2.set_xticks(sampled)
    ax2.set_xticklabels([f"{t_vals[s]:.2f}" for s in sampled], fontsize=7)
    ax2.set_xlabel("r_t  (mask ratio)", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  flow.png saved")


def _plot_entropy(
    attn_data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: str,
) -> None:
    """
    Mean attention entropy for real-query rows vs [MASK]-query rows over captured steps.

    Uses the full attention row (all S keys, not just response) so each row
    sums to ~1 and entropy is well-defined.  Lower entropy = more focused attention.
    """
    import matplotlib.pyplot as plt

    steps = sorted(attn_data.keys())
    real_ent: List[float] = []
    mask_ent: List[float] = []
    eps = 1e-12

    for step in steps:
        # Full rows: response queries attending to all S keys
        attn_rows = attn_data[step][prompt_len:, :]   # (L, S)
        mask_r = mask_data[step][prompt_len:]
        real_idx = np.where(~mask_r)[0]
        mask_idx = np.where(mask_r)[0]

        def mean_entropy(rows: np.ndarray) -> float:
            h = -np.sum(rows * np.log(rows + eps), axis=1)
            return float(h.mean())

        real_ent.append(mean_entropy(attn_rows[real_idx]) if len(real_idx) else np.nan)
        mask_ent.append(mean_entropy(attn_rows[mask_idx]) if len(mask_idx) else np.nan)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, real_ent, "o-",  color="steelblue", label="real query rows",   linewidth=1.5, markersize=4)
    ax.plot(steps, mask_ent, "s--", color="crimson",   label="[MASK] query rows", linewidth=1.5, markersize=4)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Mean row entropy (nats)")
    ax.set_title(
        "Attention entropy per token type — last layer, full key range\n"
        "Lower = more focused; TALMAS should lower real-row entropy by eliminating [MASK]-key mass"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  entropy.png saved")


def _plot_trajectory(
    mask_data: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: str,
) -> None:
    """
    Unmasking trajectory: (step × position) heatmap + first-revelation order bar.

    Top panel: red = still masked, green = already revealed at that step.
    Bottom panel: colour of each response position encodes the step at which
    it was first committed (lighter = revealed earlier).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    steps = sorted(mask_data.keys())
    L = int(mask_data[steps[0]][prompt_len:].shape[0])

    # Build (n_steps, L) grid: 1 = masked, 0 = revealed
    grid = np.stack([mask_data[s][prompt_len:].astype(np.float32) for s in steps])  # (n_steps, L)

    # First revelation step per position
    first_reveal = np.full(L, np.nan)
    prev = None
    for step in steps:
        cur = mask_data[step][prompt_len:]
        if prev is not None:
            for pos in np.where(prev & ~cur)[0]:
                if np.isnan(first_reveal[pos]):
                    first_reveal[pos] = step
        prev = cur

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )

    # Top: trajectory heatmap — position on y-axis, step on x-axis
    ax1.imshow(
        grid.T,           # (L, n_steps)
        aspect="auto",
        origin="upper",
        cmap="RdYlGn_r",  # red = masked, green = revealed
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    n_steps = len(steps)
    tick_pos = np.linspace(0, n_steps - 1, min(10, n_steps), dtype=int)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels([steps[i] for i in tick_pos], fontsize=8)
    ax1.set_ylabel("Response position")
    ax1.set_title("Unmasking trajectory  (red = masked, green = revealed)")
    ax1.tick_params(labelbottom=False)

    # Bottom: first-revelation step per position as a colour bar
    norm = mcolors.Normalize(vmin=steps[0], vmax=steps[-1])
    cmap_bar = plt.cm.plasma
    bar_vals = np.where(~np.isnan(first_reveal), first_reveal, float(steps[-1]))
    colors = cmap_bar(norm(bar_vals))
    ax2.bar(np.arange(L), 1, color=colors, width=1.0, align="center")
    ax2.set_xlim(-0.5, L - 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Response position")
    ax2.set_title("Step at which each position was first revealed  (lighter = earlier)", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, label="First revelation step",
                 orientation="horizontal", fraction=0.04, pad=0.45)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  trajectory.png saved")


def _plot_confidence_distribution(
    conf_data: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: str,
) -> None:
    """
    Violin distribution of confidence values across still-masked positions at each
    captured step.  Unmasked positions (clamped to 1.0) are excluded so the plot
    reflects the model's actual uncertainty about uncommitted tokens.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    steps = sorted(conf_data.keys())
    violin_data, violin_steps = [], []

    for step in steps:
        mask_r = mask_data[step][prompt_len:]
        masked_confs = conf_data[step][mask_r]
        if len(masked_confs) >= 2:          # violinplot requires ≥ 2 points
            violin_data.append(masked_confs)
            violin_steps.append(step)

    if not violin_data:
        return

    step_span = violin_steps[-1] - violin_steps[0] if len(violin_steps) > 1 else 1
    width = max(1, step_span / (len(violin_steps) + 1))

    fig, ax = plt.subplots(figsize=(14, 4))
    parts = ax.violinplot(
        violin_data,
        positions=violin_steps,
        widths=width,
        showmedians=True,
        showextrema=True,
    )

    norm = Normalize(vmin=violin_steps[0], vmax=violin_steps[-1])
    cmap = cm.plasma
    for pc, step in zip(parts["bodies"], violin_steps):
        pc.set_facecolor(cmap(norm(step)))
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Confidence  (still-masked positions only)")
    ax.set_title(
        "Confidence distribution for still-masked response tokens\n"
        "White bar = median; wider violin = more spread"
    )
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Denoising step", fraction=0.02, pad=0.01)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  conf_dist.png saved  ({len(violin_steps)} violins)")


def _plot_flip_rate(
    pred_all: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: str,
) -> None:
    """
    Fraction of still-masked positions whose greedy prediction changed since
    the previous step.  Only positions masked at both consecutive steps are counted.

    Lower flip rate = the model's tentative predictions are more stable,
    suggesting it is less confused by the evolving [MASK] context.
    """
    import matplotlib.pyplot as plt

    steps = sorted(pred_all.keys())
    flip_steps: List[int] = []
    flip_rates: List[float] = []

    for i in range(1, len(steps)):
        prev_s, curr_s = steps[i - 1], steps[i]
        both_masked = mask_data[prev_s][prompt_len:] & mask_data[curr_s][prompt_len:]
        if both_masked.any():
            changed = pred_all[prev_s][both_masked] != pred_all[curr_s][both_masked]
            flip_rates.append(float(changed.mean()))
        else:
            flip_rates.append(0.0)
        flip_steps.append(curr_s)

    if not flip_steps:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(flip_steps, flip_rates, "o-", color="#E87B4C", linewidth=1.5, markersize=3)
    ax.fill_between(flip_steps, flip_rates, alpha=0.15, color="#E87B4C")
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Flip rate")
    ax.set_ylim(0, min(1.0, max(flip_rates) * 1.3 + 0.05) if flip_rates else 1.0)
    ax.set_title(
        "Prediction flip rate — fraction of still-masked positions whose greedy prediction changed\n"
        "Lower = more stable tentative predictions"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  flip_rate.png saved")


def _plot_revelation_confidence(
    conf_all: Dict[int, np.ndarray],
    mask_data: Dict[int, np.ndarray],
    prompt_len: int,
    out_path: str,
) -> None:
    """
    For each response position, record the model's confidence at the exact step
    when it was first committed (i.e. confidence BEFORE the clamping that sets
    already-unmasked positions to 1.0).

    Left panel: scatter over response positions, coloured by revelation step.
    Right panel: histogram of revelation confidence values.

    Higher values = model was more certain when it committed to that token.
    """
    import matplotlib.pyplot as plt

    steps = sorted(conf_all.keys())
    rev_conf: Dict[int, float] = {}   # pos → confidence at decision step
    rev_step: Dict[int, int]  = {}    # pos → step when revealed

    prev_mask_r = None
    prev_step = None
    for step in steps:
        mask_r = mask_data[step][prompt_len:]
        if prev_mask_r is not None:
            # Positions newly revealed after prev_step's top-k selection
            # conf_all[prev_step] has unclamped confidence (pos was masked then)
            for pos in np.where(prev_mask_r & ~mask_r)[0]:
                if pos not in rev_conf:
                    rev_conf[int(pos)] = float(conf_all[prev_step][int(pos)])
                    rev_step[int(pos)] = prev_step
        prev_mask_r = mask_r.copy()
        prev_step = step

    if not rev_conf:
        return

    positions = np.array(sorted(rev_conf.keys()))
    confs     = np.array([rev_conf[p] for p in positions])
    rsteps    = np.array([rev_step[p] for p in positions])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    sc = ax1.scatter(
        positions, confs, c=rsteps,
        cmap="plasma", s=25, alpha=0.85, edgecolors="none",
        vmin=steps[0], vmax=steps[-1],
    )
    median_c = float(np.median(confs))
    ax1.axhline(median_c, color="white", linewidth=1.5, linestyle="--")
    ax1.axhline(median_c, color="gray",  linewidth=0.9, linestyle="--",
                label=f"Median = {median_c:.3f}")
    ax1.set_xlabel("Response position")
    ax1.set_ylabel("Confidence at decision step")
    ax1.set_title("Model confidence when committing to each token\n(color = denoising step)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    fig.colorbar(sc, ax=ax1, label="Revelation step")

    ax2.hist(confs, bins=min(20, len(confs)), color="#4472C4", alpha=0.85, edgecolor="white")
    ax2.axvline(np.median(confs), color="crimson",    linestyle="--", linewidth=1.5,
                label=f"Median = {np.median(confs):.3f}")
    ax2.axvline(np.mean(confs),   color="darkorange", linestyle=":",  linewidth=1.5,
                label=f"Mean   = {np.mean(confs):.3f}")
    ax2.set_xlabel("Confidence at revelation")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of revelation confidence values")
    ax2.set_xlim(0, 1)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  revelation_conf.png saved")
