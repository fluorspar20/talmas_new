"""
Unified GSM8K evaluation CLI — baseline and TALMAS mode.

Baseline usage (identical to llada_gsm8k_eval.py):
  python scripts/gsm8k_eval.py --model GSAI-ML/LLaDA-8B-Base --max_samples 100

TALMAS usage:
  python scripts/gsm8k_eval.py \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --talmas \\
      --lambda-max 4.0 \\
      --mu 0.1 \\
      --max_samples 100

Ablation (all 5 configs + μ sweep):
  python scripts/run_ablation.py --max-samples 100

Requirements:
  pip install torch transformers datasets accelerate tqdm
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import SamplingConfig, TALMASConfig, BASE_CONFIG, INSTRUCT_CONFIG
from src.utils import (
    build_prompt,
    extract_answer,
    answers_match,
    resolve_special_tokens,
    load_model_and_tokenizer,
)
from src.sampling import low_confidence_remasking_sample
from src.talmas import TALMASHookManager


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args) -> float:
    is_instruct = "instruct" in args.model.lower()
    cfg = INSTRUCT_CONFIG if is_instruct else BASE_CONFIG

    if args.generation_length:
        cfg.generation_length = args.generation_length
    if args.steps:
        cfg.steps = args.steps

    # TALMAS config
    talmas_cfg: Optional[TALMASConfig] = None
    if args.talmas:
        talmas_cfg = TALMASConfig(
            lambda_max=args.lambda_max,
            mu=args.mu,
            use_timestep_gate=not args.no_timestep_gate,
            use_layer_gate=not args.no_layer_gate,
        )

    print(f"Model:             {args.model}")
    print(f"Mode:              {'Instruct' if is_instruct else 'Base'}")
    print(f"Generation length: {cfg.generation_length}")
    print(f"Sampling steps:    {cfg.steps}")
    print(f"Zero EOS conf:     {cfg.zero_eos_confidence}")
    print(f"Samples:           {args.max_samples or 'all'}")
    if talmas_cfg:
        print(f"TALMAS:            λ_max={talmas_cfg.lambda_max}  μ={talmas_cfg.mu}  "
              f"timestep_gate={talmas_cfg.use_timestep_gate}  "
              f"layer_gate={talmas_cfg.use_layer_gate}")
    else:
        print("TALMAS:            disabled (baseline)")
    print()

    # ------------------------------------------------------------------ #
    # Load model                                                           #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Force eager attention when TALMAS is enabled so the F.sdpa monkey-patch
    # fires.  Flash Attention 2 bypasses F.scaled_dot_product_attention and
    # would silently skip TALMAS without this flag.
    eager_attn = args.talmas
    tokenizer, model = load_model_and_tokenizer(args.model, eager_attn=eager_attn)

    mask_token_id, eos_token_id = resolve_special_tokens(tokenizer, model)
    print(f"mask_token_id={mask_token_id}, eos_token_id={eos_token_id}\n")

    # ------------------------------------------------------------------ #
    # Load dataset                                                         #
    # ------------------------------------------------------------------ #
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Evaluating on {len(dataset)} examples...\n")

    # ------------------------------------------------------------------ #
    # Set up TALMAS hooks                                                  #
    # ------------------------------------------------------------------ #
    hook_manager = None
    if talmas_cfg is not None and talmas_cfg.lambda_max > 0.0:
        hook_manager = TALMASHookManager(model, talmas_cfg)

    # ------------------------------------------------------------------ #
    # Eval loop                                                            #
    # ------------------------------------------------------------------ #
    correct = 0
    total = 0
    results = []

    try:
        for example in tqdm(dataset, desc="GSM8K"):
            question  = example["question"]
            gold_full = example["answer"]
            gold_ans  = extract_answer(gold_full)

            prompt_text = build_prompt(question, is_instruct)
            prompt_ids  = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(device)

            output_ids = low_confidence_remasking_sample(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                cfg=cfg,
                device=device,
                mask_token_id=mask_token_id,
                eos_token_id=eos_token_id,
                hook_manager=hook_manager,
            )

            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            pred_ans    = extract_answer(output_text)
            is_correct  = answers_match(pred_ans, gold_ans)

            correct += int(is_correct)
            total   += 1

            results.append({
                "question":   question,
                "gold":       gold_ans,
                "prediction": pred_ans,
                "output":     output_text,
                "correct":    is_correct,
            })

            if args.verbose:
                status = "✓" if is_correct else "✗"
                print(f"\n[{total}] {status}")
                print(f"  Q: {question[:80]}...")
                print(f"  Gold: {gold_ans}  |  Pred: {pred_ans}")
                print(f"  Output: {output_text[:200]}")
    finally:
        if hook_manager is not None:
            hook_manager.remove()

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"GSM8K Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*50}")
    print(f"\nPaper reports: 70.3% (Base, 4-shot) / 69.4% (Instruct)")

    # ------------------------------------------------------------------ #
    # Save results                                                         #
    # ------------------------------------------------------------------ #
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = "talmas" if args.talmas else "baseline"
        out_path = os.path.join(args.output_dir, f"gsm8k_{tag}_{ts}.json")
    else:
        out_path = args.output_file

    if out_path:
        payload = {
            "model":       args.model,
            "accuracy":    accuracy,
            "correct":     correct,
            "total":       total,
            "sampling":    cfg.__dict__,
            "talmas":      talmas_cfg.__dict__ if talmas_cfg else None,
            "results":     results,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLaDA GSM8K evaluation — baseline and TALMAS mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model / dataset ---
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Base",
                        help="HuggingFace model name or local path")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"],
                        help="GSM8K split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of examples (None = full 1319-example test set)")
    parser.add_argument("--generation_length", type=int, default=None,
                        help="Override generation length")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override number of diffusion steps")

    # --- Output ---
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results to this specific JSON path")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to auto-name and save results JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example predictions")

    # --- TALMAS ---
    talmas = parser.add_argument_group("TALMAS options")
    talmas.add_argument("--talmas", action="store_true",
                        help="Enable TALMAS attention suppression")
    talmas.add_argument("--lambda-max", type=float, default=4.0,
                        help="λ_max: maximum logit suppression magnitude")
    talmas.add_argument("--mu", type=float, default=0.1,
                        help="μ: mask→mask suppression scale (0=full, 1=same as real→mask)")
    talmas.add_argument("--no-timestep-gate", action="store_true",
                        help="Disable f(1-r_t) quadratic timestep gate")
    talmas.add_argument("--no-layer-gate", action="store_true",
                        help="Disable g(ℓ/L) sigmoid layer gate")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)
