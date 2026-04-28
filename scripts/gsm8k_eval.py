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
            sigmoid_slope=args.sigmoid_slope,
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

    # Force eager attention when TALMAS is enabled so the block-level patch
    # fires correctly.  Flash Attention 2 bypasses the attention_bias path
    # and would silently skip TALMAS without this flag.
    eager_attn = args.talmas
    tokenizer, model = load_model_and_tokenizer(args.model, eager_attn=eager_attn)

    print(f"attn_implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")

    # Explicitly nullify flash_attn_func on every block so _scaled_dot_product_attention
    # falls through to the F.sdpa path, which respects the attention_bias we inject.
    if args.talmas:
        disabled = 0
        for name, module in model.named_modules():
            if hasattr(module, "flash_attn_func"):
                module.flash_attn_func = None
                disabled += 1
        print(f"Flash attention disabled in {disabled} blocks")

    mask_token_id, eos_token_id = resolve_special_tokens(tokenizer, model)
    print(f"mask_token_id={mask_token_id}, eos_token_id={eos_token_id}\n")

    # ------------------------------------------------------------------ #
    # Load dataset                                                         #
    # ------------------------------------------------------------------ #
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # ------------------------------------------------------------------ #
    # Resume from checkpoint if present                                    #
    # ------------------------------------------------------------------ #
    results: list = []
    correct = 0

    if args.checkpoint:
        os.makedirs(os.path.dirname(os.path.abspath(args.checkpoint)), exist_ok=True)

    if args.checkpoint and os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        correct = sum(r["correct"] for r in results)
        n_done  = len(results)
        print(f"Resuming from checkpoint '{args.checkpoint}': "
              f"{n_done} examples already done ({correct}/{n_done} correct).")
        dataset = dataset.select(range(n_done, len(dataset)))

    total = len(results)
    print(f"Evaluating on {len(dataset)} remaining examples...\n")

    # ------------------------------------------------------------------ #
    # Set up TALMAS hooks                                                  #
    # ------------------------------------------------------------------ #
    hook_manager = None
    if talmas_cfg is not None and talmas_cfg.lambda_max > 0.0:
        hook_manager = TALMASHookManager(model, talmas_cfg)

    # ------------------------------------------------------------------ #
    # Eval loop                                                            #
    # ------------------------------------------------------------------ #
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

            entry = {
                "question":   question,
                "gold":       gold_ans,
                "prediction": pred_ans,
                "output":     output_text,
                "correct":    is_correct,
            }
            results.append(entry)

            # Append to checkpoint immediately so progress survives preemption
            if args.checkpoint:
                with open(args.checkpoint, "a") as ckpt_f:
                    ckpt_f.write(json.dumps(entry) + "\n")

            status = "✓" if is_correct else "✗"
            running_acc = correct / total * 100
            tqdm.write(
                f"[{total:>4}] {status}  gold={str(gold_ans):<8}  pred={str(pred_ans):<8}  "
                f"running acc: {correct}/{total} ({running_acc:.1f}%)"
            )
            if args.verbose:
                tqdm.write(f"       Q: {question[:80]}...")
                tqdm.write(f"       Output: {output_text[:200]}")
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
    parser.add_argument("--generation_length", type=int, default=256,
                        help="Number of response tokens to generate")
    parser.add_argument("--steps", type=int, default=256,
                        help="Number of diffusion steps")

    # --- Output ---
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save results to this specific JSON path")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to auto-name and save results JSON")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="JSONL file for incremental checkpointing; resumes from this "
                             "file if it already exists (safe to reuse on preemption)")
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
    talmas.add_argument("--sigmoid-slope", type=float, default=8.0,
                        help="Steepness of the sigmoid layer gate g(ℓ/L)")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)