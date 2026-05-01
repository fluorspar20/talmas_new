"""
TALMAS diagnostics visualization — runs one GSM8K example and saves plots.

Outputs written to --output-dir:
  attention.png    — grid of attention heatmaps (last layer, every 10 steps)
  suppression.png  — grid of suppression bias heatmaps (last layer, every 10 steps) [TALMAS only]
  confidence.png   — token confidence per position at every 10 steps
  scalar.png       — mean attention-to-[MASK] over captured denoising steps

Usage:
  # Baseline (no suppression):
  python scripts/visualize.py \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --index 0 --steps 64 --output-dir results/viz_baseline

  # With TALMAS:
  python scripts/visualize.py \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --index 0 --steps 64 \\
      --talmas --lambda-max 4.0 --mu 0.1 \\
      --output-dir results/viz_talmas

  # Verify suppression numerically (prints before/after attention tables):
  python scripts/visualize.py \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --index 0 --steps 64 \\
      --talmas --lambda-max 4.0 --mu 0.1 \\
      --log-suppression --output-dir results/viz_talmas

Run both and compare scalar.png to see whether suppression is reducing
attention to [MASK] tokens.  Add --log-suppression to print numerical
before-vs-after tables at the first, middle, and last diffusion steps.
"""

import dataclasses
import sys
import os
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

from src.config import SamplingConfig, TALMASConfig, INSTRUCT_CONFIG, BASE_CONFIG
from src.utils import build_prompt, resolve_special_tokens, load_model_and_tokenizer
from src.sampling import low_confidence_remasking_sample
from src.talmas import TALMASHookManager
from src.diagnostics import DiagnosticsCollector
from src.suppression_log import SuppresssionLogger


def main(args) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    is_instruct = "instruct" in args.model.lower()
    cfg = INSTRUCT_CONFIG if is_instruct else BASE_CONFIG
    overrides = {}
    if args.steps:
        overrides["steps"] = args.steps
    if args.generation_length:
        overrides["generation_length"] = args.generation_length
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    talmas_cfg = None
    if args.talmas:
        talmas_cfg = TALMASConfig(
            lambda_max=args.lambda_max,
            mu=args.mu,
            use_timestep_gate=not args.no_timestep_gate,
            use_layer_gate=not args.no_layer_gate,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if talmas_cfg:
        print(f"TALMAS: λ_max={talmas_cfg.lambda_max}  μ={talmas_cfg.mu}  "
              f"timestep_gate={talmas_cfg.use_timestep_gate}  "
              f"layer_gate={talmas_cfg.use_layer_gate}")
    else:
        print("TALMAS: disabled (baseline)")

    tokenizer, model = load_model_and_tokenizer(args.model, eager_attn=True)

    disabled = sum(
        1 for _, m in model.named_modules()
        if hasattr(m, "flash_attn_func") and setattr(m, "flash_attn_func", None) is None
    )
    if disabled:
        print(f"Flash attention disabled in {disabled} blocks")

    mask_token_id, eos_token_id = resolve_special_tokens(tokenizer, model)
    num_layers = model.config.num_hidden_layers
    print(f"mask_token_id={mask_token_id}  eos_token_id={eos_token_id}  layers={num_layers}")

    # ------------------------------------------------------------------ #
    # Load example                                                         #
    # ------------------------------------------------------------------ #
    print("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    example = dataset[args.index]
    question = example["question"]
    gold = example["answer"].split("####")[-1].strip()
    print(f"Example [{args.index}]: {question[:100]}...")

    prompt_text = build_prompt(question, is_instruct)
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    print(f"Prompt: {prompt_len} tokens  |  Gen: {cfg.generation_length}  |  Steps: {cfg.steps}")

    # ------------------------------------------------------------------ #
    # Install hooks  (order matters — each wraps the previous)            #
    # ------------------------------------------------------------------ #
    hook_manager = None
    if talmas_cfg is not None and talmas_cfg.lambda_max > 0.0:
        hook_manager = TALMASHookManager(model, talmas_cfg)  # 1st

    # SuppresssionLogger sits between TALMAS and DiagnosticsCollector so
    # the F.sdpa call chain for the last block is:
    #   DiagnosticsCollector → SuppresssionLogger → real F.sdpa
    # Both loggers see Q / K / attn_mask independently.
    supp_logger = None
    if args.log_suppression:
        supp_logger = SuppresssionLogger(                             # 2nd
            model,
            total_steps=cfg.steps,
            n_samples=args.log_samples,
        )
        print(f"SuppresssionLogger: active  (n_samples={args.log_samples} per type, "
              f"steps 0 / {cfg.steps // 2} / {cfg.steps - 1})")

    # DiagnosticsCollector is outermost so it wraps the already-patched forward.
    diagnostics = DiagnosticsCollector(                               # 3rd
        model, talmas_cfg, num_layers,
        capture_attn_every=args.capture_attn_every,
        capture_conf_every=args.capture_conf_every,
    )

    # ------------------------------------------------------------------ #
    # Generate                                                             #
    # ------------------------------------------------------------------ #
    print("\nRunning denoising loop...")
    try:
        output_ids = low_confidence_remasking_sample(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            cfg=cfg,
            device=device,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            hook_manager=hook_manager,
            diagnostics=diagnostics,
            supp_logger=supp_logger,
        )
    finally:
        # Remove in reverse install order
        diagnostics.remove()                     # 3rd installed, 1st removed
        if supp_logger is not None:
            supp_logger.remove()                 # 2nd installed, 2nd removed
        if hook_manager is not None:
            hook_manager.remove()                # 1st installed, 3rd removed

    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"\nGenerated:\n{output_text[:500]}")
    print(f"\nExpected answer: {gold}")

    # ------------------------------------------------------------------ #
    # Plot                                                                  #
    # ------------------------------------------------------------------ #
    diagnostics.plot_all(prompt_len, args.output_dir)
    print(f"\nDone. Open {args.output_dir}/ to view plots.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TALMAS diagnostics visualization on one GSM8K example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--index", type=int, default=0,
                   help="GSM8K test set example index")
    p.add_argument("--steps", type=int, default=64,
                   help="Diffusion steps (keep ≤64 for speed; full eval uses 256)")
    p.add_argument("--generation-length", type=int, default=None, dest="generation_length",
                   help="Override generation length (default: model preset)")
    p.add_argument("--output-dir", type=str, default="results/viz", dest="output_dir")
    p.add_argument("--capture-attn-every", type=int, default=10, dest="capture_attn_every",
                   help="Capture attention/suppression every N steps (GIF frames)")
    p.add_argument("--capture-conf-every", type=int, default=10, dest="capture_conf_every",
                   help="Record confidence every N steps (lines in confidence.png)")

    talmas = p.add_argument_group("TALMAS options")
    talmas.add_argument("--talmas", action="store_true",
                        help="Enable TALMAS suppression")
    talmas.add_argument("--lambda-max", type=float, default=4.0, dest="lambda_max")
    talmas.add_argument("--mu", type=float, default=0.1)
    talmas.add_argument("--no-timestep-gate", action="store_true", dest="no_timestep_gate")
    talmas.add_argument("--no-layer-gate", action="store_true", dest="no_layer_gate")

    log = p.add_argument_group("Suppression logging options")
    log.add_argument("--log-suppression", action="store_true", dest="log_suppression",
                     help="Print before-vs-after attention tables at first, middle, and "
                          "last diffusion steps to verify suppression is active.  "
                          "Works with or without --talmas (baseline shows Δ=0).")
    log.add_argument("--log-samples", type=int, default=3, dest="log_samples",
                     help="Token pairs to sample per interaction type in each table "
                          "(real→real, real→[MASK], [MASK]→real, [MASK]→[MASK])")

    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
