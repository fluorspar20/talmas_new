"""
LLaDA reverse-diffusion sampling  (Algorithm 5: low-confidence remasking).

Extracted verbatim from llada_gsm8k_eval.py with one addition: an optional
hook_manager parameter that updates TALMAS state before each forward pass.
All Algorithm 5 logic is unchanged.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.config import SamplingConfig


@torch.inference_mode()
def low_confidence_remasking_sample(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (1, prompt_len)  — already on device
    cfg: SamplingConfig,
    device: torch.device,
    mask_token_id: int,
    eos_token_id: int,
    hook_manager=None,                 # TALMASHookManager | None
) -> torch.Tensor:
    """
    Pure diffusion sampling with low-confidence remasking (Algorithm 5).

    If hook_manager is provided, its state is updated at each diffusion step
    so that TALMAS logit biases are applied during the forward pass.

    Returns the generated token ids as a 1-D tensor (response only).
    """
    L = cfg.generation_length
    N = cfg.steps

    # ------------------------------------------------------------------ #
    # Step 1 — initialise: fully masked response appended to prompt        #
    # ------------------------------------------------------------------ #
    response = torch.full((1, L), mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, response], dim=1)  # (1, prompt_len + L)
    prompt_len = prompt_ids.shape[1]

    # uniform time-steps: t goes from 1 → 1/N in steps of 1/N
    timesteps = torch.linspace(1.0, 1.0 / N, N, device=device)

    for i, t in enumerate(timesteps):
        s = t - 1.0 / N          # next time-step (can be 0 at last step)
        s = max(s.item(), 0.0)
        t_val = t.item()

        # -------------------------------------------------------------- #
        # TALMAS: update hook state with current mask ratio and positions  #
        # t_val IS r_t (fraction of tokens still masked at this step)      #
        # -------------------------------------------------------------- #
        if hook_manager is not None:
            mask_positions = (input_ids == mask_token_id)  # (1, prompt_len+L)
            hook_manager.set_state(r_t=t_val, mask_positions=mask_positions)

        # -------------------------------------------------------------- #
        # Step 2 — forward pass: predict all masked tokens simultaneously #
        # -------------------------------------------------------------- #
        logits = model(input_ids=input_ids).logits  # (1, prompt_len+L, vocab)
        response_logits = logits[0, prompt_len:, :]  # (L, vocab)

        probs = F.softmax(response_logits, dim=-1)   # (L, vocab)

        # Greedy prediction and confidence score for each position
        pred_ids   = probs.argmax(dim=-1)            # (L,)
        confidence = probs.max(dim=-1).values        # (L,)

        # -------------------------------------------------------------- #
        # Optional: zero out EOS confidence (instruct model only)         #
        # -------------------------------------------------------------- #
        if cfg.zero_eos_confidence:
            eos_mask = (pred_ids == eos_token_id)
            confidence = confidence.masked_fill(eos_mask, 0.0)

        # -------------------------------------------------------------- #
        # Step 3 — decide which positions to keep vs remask               #
        # -------------------------------------------------------------- #
        # Positions that were already unmasked stay unmasked (confidence=1)
        current_response = input_ids[0, prompt_len:]  # (L,)
        already_unmasked  = (current_response != mask_token_id)
        confidence = confidence.masked_fill(already_unmasked, 1.0)
        # TODO: Print confidence values for every token for every 10 denoising steps
        pred_ids   = torch.where(already_unmasked, current_response, pred_ids)

        # Number of tokens that should be unmasked at time s
        n_unmask = math.floor(L * (1.0 - s))
        n_unmask = max(0, min(n_unmask, L))

        # Select the n_unmask positions with the highest confidence.
        # stable=True breaks ties by input order, eliminating non-determinism
        # from bfloat16 probability ties across runs.
        _, top_indices = torch.topk(confidence, k=n_unmask, largest=True, stable=True)
        new_response = torch.full((L,), mask_token_id, dtype=torch.long, device=device)
        new_response[top_indices] = pred_ids[top_indices]

        input_ids[0, prompt_len:] = new_response

    # Final response (drop everything after the first EOS if present)
    final = input_ids[0, prompt_len:]
    eos_positions = (final == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        final = final[: eos_positions[0]]

    return final
