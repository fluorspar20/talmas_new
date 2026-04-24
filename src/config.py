"""
Shared configuration dataclasses and presets for LLaDA evaluation and TALMAS ablation.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Sampling config  (verbatim from llada_gsm8k_eval.py)
# ---------------------------------------------------------------------------

@dataclass
class SamplingConfig:
    generation_length: int       # number of response tokens to generate
    steps: int                   # number of reverse diffusion steps (N)
    zero_eos_confidence: bool    # set EOS confidence=0 (needed for instruct model)
    few_shot: int = 4            # number of few-shot examples


BASE_CONFIG = SamplingConfig(
    generation_length=1024,
    steps=1024,
    zero_eos_confidence=False,
)

INSTRUCT_CONFIG = SamplingConfig(
    generation_length=512,
    steps=512,
    zero_eos_confidence=True,
)


# ---------------------------------------------------------------------------
# TALMAS config
# ---------------------------------------------------------------------------

@dataclass
class TALMASConfig:
    lambda_max: float = 4.0          # maximum logit suppression magnitude
    mu: float = 0.1                  # mask→mask partial suppression scale
    use_timestep_gate: bool = True   # apply quadratic f(1-r_t) gate
    use_layer_gate: bool = True      # apply sigmoid g(ℓ/L) layer gate


# ---------------------------------------------------------------------------
# Ablation configurations  (5 configs from TALMAS proposal)
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    {
        "id": 1,
        "name": "Baseline (LLaDA)",
        "lambda_max": 0.0,
        "use_timestep_gate": False,
        "use_layer_gate": False,
        "mu": 0.0,
        "description": "λ_max=0 — recovers original LLaDA exactly",
    },
    {
        "id": 2,
        "name": "Static Bias",
        "lambda_max": 4.0,
        "use_timestep_gate": False,
        "use_layer_gate": False,
        "mu": 0.0,
        "description": "Fixed λ, no timestep or layer gating",
    },
    {
        "id": 3,
        "name": "Timestep-Only",
        "lambda_max": 4.0,
        "use_timestep_gate": True,
        "use_layer_gate": False,
        "mu": 0.0,
        "description": "Quadratic timestep ramp, uniform across layers",
    },
    {
        "id": 4,
        "name": "Layer-Only",
        "lambda_max": 4.0,
        "use_timestep_gate": False,
        "use_layer_gate": True,
        "mu": 0.0,
        "description": "Sigmoid layer gate, uniform across timesteps",
    },
    {
        "id": 5,
        "name": "Full TALMAS",
        "lambda_max": 4.0,
        "use_timestep_gate": True,
        "use_layer_gate": True,
        "mu": 0.1,
        "description": "Full joint gating with partial mask→mask suppression",
    },
]

MU_SWEEP = [0.0, 0.2, 0.5, 1.0]
