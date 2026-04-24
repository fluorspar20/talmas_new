"""
TALMAS: Timestep-Adaptive, Layer-Dependent Masked Attention Suppression.

Implements the gate functions and hook manager described in the TALMAS proposal.
The hook manager monkey-patches each attention layer's forward() to inject a
logit bias before the softmax — no model weights are modified.

Suppression regime:
  Query \ Key  |  Real (m_j=0)  |  [MASK] (m_j=1)
  -------------|----------------|------------------
  Real (m_i=0) |  no bias       |  -λ         (full suppression)
  [MASK](m_i=1)|  no bias       |  -λ·μ       (partial; preserves coordination)

λ(ℓ, t) = λ_max · f(1-r_t) · g(ℓ/L)
  f(x) = x²               (timestep gate: grows as more tokens are revealed)
  g(u) = sigmoid(8(u-0.5)) (layer gate: peaks at deep layers)
"""

import functools
from typing import Optional

import torch
import torch.nn.functional as F

from src.config import TALMASConfig


# ---------------------------------------------------------------------------
# Gate functions
# ---------------------------------------------------------------------------

def f_timestep(x: float) -> float:
    """Quadratic timestep gate.  x = 1 - r_t (fraction of tokens revealed)."""
    return x ** 2


def g_layer(layer_idx: int, num_layers: int) -> float:
    """Sigmoid layer gate.  Peaks at deep layers."""
    u = layer_idx / num_layers
    return torch.sigmoid(torch.tensor(8.0 * (u - 0.5))).item()


def compute_lambda(
    lambda_max: float,
    r_t: float,
    layer_idx: int,
    num_layers: int,
    use_timestep_gate: bool = True,
    use_layer_gate: bool = True,
) -> float:
    """
    λ(ℓ, t) = λ_max · f(1-r_t) · g(ℓ/L)

    r_t: current mask ratio (fraction of tokens still masked), 0 ≤ r_t ≤ 1.
    Gates can be disabled independently for ablation.
    """
    x = 1.0 - r_t  # fraction revealed
    f = f_timestep(x) if use_timestep_gate else 1.0
    g = g_layer(layer_idx, num_layers) if use_layer_gate else 1.0
    return lambda_max * f * g


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class TALMASHookManager:
    """
    Registers patched forward methods on each self-attention layer to inject
    the TALMAS asymmetric logit bias before softmax.

    Usage:
        manager = TALMASHookManager(model, cfg)
        for step in denoising_loop:
            manager.set_state(r_t=..., mask_positions=...)
            output = model(input_ids=...)
        manager.remove()
    """

    def __init__(self, model, cfg: TALMASConfig):
        self.model = model
        self.cfg = cfg

        # Runtime state — updated each diffusion step via set_state()
        self.r_t: float = 1.0
        self.mask_positions: Optional[torch.Tensor] = None  # (batch, seq_len) bool

        self._patched = []                              # list of (module, original_forward)
        self.num_layers = self._count_layers()
        self._register_patches()

    def _count_layers(self) -> int:
        if hasattr(self.model.config, "num_hidden_layers"):
            return self.model.config.num_hidden_layers
        # Fallback: count self_attn modules
        count = sum(1 for n, _ in self.model.named_modules() if n.endswith(".self_attn"))
        return max(count, 1)

    def _patch_attention(self, attn_module, layer_idx: int) -> None:
        """Replace attn_module.forward so TALMAS bias is injected via F.sdpa."""
        manager = self
        original_sdpa = F.scaled_dot_product_attention
        original_forward = attn_module.forward

        @functools.wraps(original_sdpa)
        def patched_sdpa(query, key, value, attn_mask=None, **kw):
            # Skip if hook not armed or bias is zero
            if (
                manager.mask_positions is None
                or manager.cfg.lambda_max == 0.0
            ):
                return original_sdpa(query, key, value, attn_mask=attn_mask, **kw)

            # Stale-state guard: mask_positions must match sequence length
            seq_len = query.shape[-2]
            if manager.mask_positions.shape[-1] != seq_len:
                return original_sdpa(query, key, value, attn_mask=attn_mask, **kw)

            lam = compute_lambda(
                manager.cfg.lambda_max,
                manager.r_t,
                layer_idx,
                manager.num_layers,
                use_timestep_gate=manager.cfg.use_timestep_gate,
                use_layer_gate=manager.cfg.use_layer_gate,
            )

            if lam == 0.0:
                return original_sdpa(query, key, value, attn_mask=attn_mask, **kw)

            # Build bias  (B, 1, S, S) — broadcasts over heads
            m = manager.mask_positions.float().to(query.device)  # (B, S)
            m_key   = m.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
            m_query = m.unsqueeze(1).unsqueeze(3)   # (B, 1, S, 1)
            query_gate = (1.0 - m_query) + manager.cfg.mu * m_query  # (B, 1, S, 1)
            bias = -(lam * m_key * query_gate)       # (B, 1, S, S)

            combined_mask = bias if attn_mask is None else attn_mask + bias
            return original_sdpa(query, key, value, attn_mask=combined_mask, **kw)

        # Store patch reference on the module so it isn't GC'd
        attn_module._talmas_patched_sdpa = patched_sdpa

        @functools.wraps(original_forward)
        def patched_forward(*args, **kwargs):
            old = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = attn_module._talmas_patched_sdpa
            try:
                return original_forward(*args, **kwargs)
            finally:
                F.scaled_dot_product_attention = old

        attn_module.forward = patched_forward
        self._patched.append((attn_module, original_forward))

    def _register_patches(self) -> None:
        layer_idx = 0
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn"):
                self._patch_attention(module, layer_idx)
                layer_idx += 1

    def set_state(self, r_t: float, mask_positions: torch.Tensor) -> None:
        """Call once per diffusion step before the forward pass."""
        self.r_t = r_t
        self.mask_positions = mask_positions

    def remove(self) -> None:
        """Restore all original attention forward methods."""
        for attn_module, original_forward in self._patched:
            attn_module.forward = original_forward
        self._patched.clear()
        self.mask_positions = None
