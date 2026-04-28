"""
TALMAS: Timestep-Adaptive, Layer-Dependent Masked Attention Suppression.

Implements the gate functions and hook manager described in the TALMAS proposal.
The hook manager patches each LLaDALlamaBlock's forward() to inject a logit bias
into attention_bias before the attention call — no model weights are modified.

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

def f_timestep(x: float, exponent: float = 2.0) -> float:
    """Timestep gate.  x = 1 - r_t (fraction of tokens revealed)."""
    return x ** exponent


def g_layer(layer_idx: int, num_layers: int, slope: float = 8.0) -> float:
    """Sigmoid layer gate.  Peaks at deep layers."""
    u = layer_idx / num_layers
    return torch.sigmoid(torch.tensor(slope * (u - 0.5))).item()


def compute_lambda(
    lambda_max: float,
    r_t: float,
    layer_idx: int,
    num_layers: int,
    use_timestep_gate: bool = True,
    use_layer_gate: bool = True,
    sigmoid_slope: float = 8.0,
    timestep_exponent: float = 2.0,
) -> float:
    """
    λ(ℓ, t) = λ_max · f(1-r_t) · g(ℓ/L)

    r_t: current mask ratio (fraction of tokens still masked), 0 ≤ r_t ≤ 1.
    Gates can be disabled independently for ablation.
    """
    x = 1.0 - r_t  # fraction revealed
    f = f_timestep(x, exponent=timestep_exponent) if use_timestep_gate else 1.0
    g = g_layer(layer_idx, num_layers, slope=sigmoid_slope) if use_layer_gate else 1.0
    return lambda_max * f * g


# ---------------------------------------------------------------------------
# Hook manager
# ---------------------------------------------------------------------------

class TALMASHookManager:
    """
    Patches each LLaDALlamaBlock's forward() to inject the TALMAS asymmetric
    logit bias into attention_bias before the attention call.

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
        # Fallback: count transformer blocks
        count = sum(
            1 for name, _ in self.model.named_modules()
            if "transformer.blocks." in name and name.count(".") == 3
        )
        return max(count, 1)

    def _patch_attention(self, block_module, layer_idx: int) -> None:
        """
        Patch LLaDALlamaBlock.forward to inject TALMAS bias into attention_bias.
        LLaDA passes attention_bias directly into self.attention(q, k, v, attention_bias),
        so we intercept at the block level and add our logit bias there.
        """
        manager = self
        original_forward = block_module.forward

        @functools.wraps(original_forward)
        def patched_forward(x, attention_bias=None, **kwargs):
            if manager.mask_positions is not None and manager.cfg.lambda_max > 0.0:
                lam = compute_lambda(
                    manager.cfg.lambda_max,
                    manager.r_t,
                    layer_idx,
                    manager.num_layers,
                    use_timestep_gate=manager.cfg.use_timestep_gate,
                    use_layer_gate=manager.cfg.use_layer_gate,
                    sigmoid_slope=manager.cfg.sigmoid_slope,
                    timestep_exponent=manager.cfg.timestep_exponent,
                )

                if lam > 0.0:
                    m = manager.mask_positions.float().to(x.device)  # (B, S)

                    m_key   = m.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
                    m_query = m.unsqueeze(1).unsqueeze(3)   # (B, 1, S, 1)
                    query_gate = (1.0 - m_query) + manager.cfg.mu * m_query
                    talmas_bias = -(lam * m_key * query_gate)       # (B, 1, S, S)

                    # Match the dtype that _cast_attn_bias expects
                    talmas_bias = talmas_bias.to(dtype=x.dtype)

                    if attention_bias is not None:
                        attention_bias = attention_bias + talmas_bias
                    else:
                        attention_bias = talmas_bias

            return original_forward(x, attention_bias=attention_bias, **kwargs)

        block_module.forward = patched_forward
        self._patched.append((block_module, original_forward))

    def _register_patches(self) -> None:
        """Patch each LLaDALlamaBlock — named model.transformer.blocks.N"""
        layer_idx = 0
        for name, module in self.model.named_modules():
            if "transformer.blocks." in name and name.count(".") == 3:
                self._patch_attention(module, layer_idx)
                layer_idx += 1

    def set_state(self, r_t: float, mask_positions: torch.Tensor) -> None:
        """Call once per diffusion step before the forward pass."""
        self.r_t = r_t
        self.mask_positions = mask_positions

    def remove(self) -> None:
        """Restore all original block forward methods."""
        for block_module, original_forward in self._patched:
            block_module.forward = original_forward
        self._patched.clear()
        self.mask_positions = None