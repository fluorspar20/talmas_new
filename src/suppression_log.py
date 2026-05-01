"""
SuppresssionLogger: prints before-vs-after attention weight tables at three
diffusion steps (first, middle, last) to verify TALMAS suppression is active.

For each target step the last transformer layer's F.scaled_dot_product_attention
call is intercepted.  We have access to Q, K, and `attn_mask` (the TALMAS
logit bias).  From these we compute:
  before — softmax(QKᵀ / √d)               raw weights, no bias
  after  — softmax(QKᵀ / √d + attn_mask)   weights after suppression

For baseline LLaDA attn_mask is None, so before == after and Δ == 0.

Install order relative to other hooks on the last block:
  AFTER TALMASHookManager, BEFORE DiagnosticsCollector.

This ensures the F.sdpa call chain is:
  DiagnosticsCollector.capturing_sdpa
    → SuppresssionLogger.logging_sdpa      (inner old_sdpa points to capturing_sdpa)
      → real F.scaled_dot_product_attention

Both loggers see the same Q / K / attn_mask; model output is unaffected.

Remove order (reverse of install):
  DiagnosticsCollector → SuppresssionLogger → TALMASHookManager.
"""

import functools
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class SuppresssionLogger:
    """
    Logs attention weights before and after TALMAS suppression at three
    diffusion steps: first (step 0), middle (step N//2), last (step N-1).

    For each target step a table is printed with sampled token pairs from all
    four interaction types: real→real, real→[MASK], [MASK]→real, [MASK]→[MASK].
    """

    def __init__(self, model, total_steps: int, n_samples: int = 3):
        """
        Args:
            model:       LLaDA model (already passed through TALMASHookManager).
            total_steps: Total number of diffusion steps N.
            n_samples:   Pairs to sample per interaction type (default 3).
        """
        self.total_steps = total_steps
        self.n_samples = n_samples

        mid = total_steps // 2
        self._target_steps = {0, mid, total_steps - 1}
        self._step_labels = {0: "first", mid: "middle", total_steps - 1: "last"}

        self._last_block = None
        self._orig_fwd = None

        self._armed = False
        self._step_idx = -1
        self._mask_positions: Optional[torch.Tensor] = None
        self._pending: Optional[dict] = None

        self._install(model)

    # ------------------------------------------------------------------
    # Hook installation / removal
    # ------------------------------------------------------------------

    def _install(self, model) -> None:
        last = None
        for name, module in model.named_modules():
            if "transformer.blocks." in name and name.count(".") == 3:
                last = module
        if last is None:
            print("SuppresssionLogger: last transformer block not found — disabled")
            return

        self._last_block = last
        self._orig_fwd = last.forward
        logger = self

        @functools.wraps(self._orig_fwd)
        def logging_fwd(x, attention_bias=None, **kwargs):
            if not logger._armed:
                return logger._orig_fwd(x, attention_bias=attention_bias, **kwargs)

            pending: dict = {}
            # Capture whatever is currently installed as F.sdpa (may be
            # DiagnosticsCollector.capturing_sdpa if that logger is armed).
            old_sdpa = F.scaled_dot_product_attention

            def logging_sdpa(query, key, value, attn_mask=None, **kw):
                # Pass through to the outer sdpa in the chain first so that
                # DiagnosticsCollector (and ultimately the real F.sdpa) also
                # get to run — model output is determined here.
                out = old_sdpa(query, key, value, attn_mask=attn_mask, **kw)

                scale = query.shape[-1] ** -0.5
                with torch.no_grad():
                    # Mean Q·Kᵀ over heads → (S, S), float32 for precision
                    qk = torch.matmul(
                        query.float(), key.float().transpose(-2, -1)
                    ) * scale  # (B, H, S, S)

                    wt_before = (
                        torch.softmax(qk, dim=-1).mean(dim=1)[0]
                        .cpu().numpy().astype(np.float32)
                    )  # (S, S) — no bias

                    if attn_mask is not None:
                        # attn_mask contains the TALMAS logit bias for this layer
                        wt_after = (
                            torch.softmax(qk + attn_mask.float(), dim=-1)
                            .mean(dim=1)[0]
                            .cpu().numpy().astype(np.float32)
                        )  # (S, S) — suppressed
                    else:
                        wt_after = wt_before   # baseline: no suppression

                    pending["before"] = wt_before
                    pending["after"] = wt_after
                return out

            F.scaled_dot_product_attention = logging_sdpa
            try:
                result = logger._orig_fwd(x, attention_bias=attention_bias, **kwargs)
            finally:
                F.scaled_dot_product_attention = old_sdpa

            logger._pending = pending if pending else None
            return result

        last.forward = logging_fwd

    def remove(self) -> None:
        if self._last_block is not None and self._orig_fwd is not None:
            self._last_block.forward = self._orig_fwd

    # ------------------------------------------------------------------
    # Sampling loop interface
    # ------------------------------------------------------------------

    def begin_step(self, step_idx: int, mask_positions: torch.Tensor) -> None:
        """Call BEFORE the model forward pass."""
        self._armed = (step_idx in self._target_steps)
        self._step_idx = step_idx
        self._mask_positions = mask_positions
        self._pending = None

    def log(self, step_idx: int, prompt_len: int) -> None:
        """Call AFTER the model forward pass.  Prints the table for target steps."""
        self._armed = False
        if self._pending is None or self._mask_positions is None:
            return

        before = self._pending["before"]  # (S, S)
        after = self._pending["after"]    # (S, S)
        mask = self._mask_positions[0].cpu().numpy().astype(bool)  # (S,)

        # Crop both axes to the response region so indices are response-local
        br = before[prompt_len:, prompt_len:]   # (L, L)
        ar = after[prompt_len:, prompt_len:]    # (L, L)
        mr = mask[prompt_len:]                  # (L,)

        real_idx = np.where(~mr)[0]
        mask_idx = np.where(mr)[0]

        rng = np.random.default_rng(42)
        pairs = _sample_pairs(real_idx, mask_idx, self.n_samples, rng)

        label = self._step_labels.get(step_idx, str(step_idx))
        _print_table(
            step_idx=step_idx,
            step_label=label,
            L=len(mr),
            n_masked=int(mr.sum()),
            before=br,
            after=ar,
            pairs=pairs,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_QUADRANT_LABEL = {
    ("real", "real"): "real  → real  ",
    ("real", "mask"): "real  → [MASK]",
    ("mask", "real"): "[MASK]→ real  ",
    ("mask", "mask"): "[MASK]→ [MASK]",
}


def _sample_pairs(
    real_idx: np.ndarray,
    mask_idx: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> list:
    """Return (qi, ki, q_type, k_type) tuples for all available quadrants."""
    pairs = []
    for q_type, q_pool in [("real", real_idx), ("mask", mask_idx)]:
        for k_type, k_pool in [("real", real_idx), ("mask", mask_idx)]:
            if len(q_pool) == 0 or len(k_pool) == 0:
                continue
            n = min(n_samples, len(q_pool), len(k_pool))
            qi_samp = q_pool[rng.choice(len(q_pool), n, replace=False)]
            ki_samp = k_pool[rng.choice(len(k_pool), n, replace=False)]
            for qi, ki in zip(qi_samp, ki_samp):
                pairs.append((int(qi), int(ki), q_type, k_type))
    return pairs


def _print_table(
    step_idx: int,
    step_label: str,
    L: int,
    n_masked: int,
    before: np.ndarray,
    after: np.ndarray,
    pairs: list,
) -> None:
    W = 78
    print(f"\n{'═' * W}")
    print(
        f"  Suppression check — step {step_idx} ({step_label})"
        f"  |  {n_masked}/{L} response tokens masked"
        f"  |  last layer, mean over heads"
    )
    print(f"{'═' * W}")
    print(
        f"  {'Interaction type':<16}  {'Query':>6}  {'Key':>6}"
        f"  {'Before':>10}  {'After':>10}  {'Δ (After−Before)':>16}"
    )
    print(f"  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*16}")
    for qi, ki, q_type, k_type in pairs:
        b = float(before[qi, ki])
        a = float(after[qi, ki])
        lbl = _QUADRANT_LABEL[(q_type, k_type)]
        print(
            f"  {lbl:<16}  {qi:>6}  {ki:>6}"
            f"  {b:>10.6f}  {a:>10.6f}  {a - b:>+16.6f}"
        )
    print(f"{'═' * W}")
