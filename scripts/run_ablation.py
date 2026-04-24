"""
TALMAS ablation study runner.

Runs all 5 ablation configurations (plus μ sweep for Full TALMAS) on GSM8K,
saves a CSV results table, and produces ablation plots.

Usage:
  python scripts/run_ablation.py \\
      --model GSAI-ML/LLaDA-8B-Instruct \\
      --max-samples 100 \\
      --output-dir results

For a quick smoke-test (5 samples, 20 steps):
  python scripts/run_ablation.py --max-samples 5 --steps 20

Note: config 1 (λ_max=0) acts as the baseline and should match
llada_gsm8k_eval.py output exactly at the same sample count.
"""

import sys
import os
import argparse
from datetime import datetime

import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    SamplingConfig, TALMASConfig, BASE_CONFIG, INSTRUCT_CONFIG,
    ABLATION_CONFIGS, MU_SWEEP,
)
from src.utils import (
    build_prompt, extract_answer, answers_match,
    resolve_special_tokens, load_model_and_tokenizer,
)
from src.sampling import low_confidence_remasking_sample
from src.talmas import TALMASHookManager


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------

def run_one_config(
    *,
    model,
    tokenizer,
    device,
    mask_token_id: int,
    eos_token_id: int,
    cfg: SamplingConfig,
    talmas_cfg: TALMASConfig,
    is_instruct: bool,
    dataset,
    config_meta: dict,
) -> dict:
    name = config_meta["name"]
    print(f"\n{'='*60}")
    print(f"Config {config_meta['id']}: {name}")
    print(f"  {config_meta['description']}")
    print(f"  λ_max={talmas_cfg.lambda_max}  μ={talmas_cfg.mu}  "
          f"timestep_gate={talmas_cfg.use_timestep_gate}  "
          f"layer_gate={talmas_cfg.use_layer_gate}")
    print(f"{'='*60}")

    hook_manager = None
    if talmas_cfg.lambda_max > 0.0:
        hook_manager = TALMASHookManager(model, talmas_cfg)

    correct = 0
    total = 0
    raw_results = []

    try:
        for example in tqdm(dataset, desc=name):
            question  = example["question"]
            gold_ans  = extract_answer(example["answer"])
            prompt_text = build_prompt(question, is_instruct)
            prompt_ids  = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=True
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
            raw_results.append({"pred": pred_ans, "gold": gold_ans, "correct": is_correct})
    finally:
        if hook_manager is not None:
            hook_manager.remove()

    accuracy = correct / total if total > 0 else 0.0
    print(f"  → Accuracy: {accuracy:.3f}  ({correct}/{total})")

    return {
        "config_id":          config_meta["id"],
        "config_name":        name,
        "lambda_max":         talmas_cfg.lambda_max,
        "mu":                 talmas_cfg.mu,
        "use_timestep_gate":  talmas_cfg.use_timestep_gate,
        "use_layer_gate":     talmas_cfg.use_layer_gate,
        "n_samples":          total,
        "correct":            correct,
        "accuracy":           accuracy,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, mu_sweep_values: list, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: main ablation bar chart (one bar per config; use μ=0.1 row for config 5)
    main = df[~df["config_name"].str.contains(r"μ=", regex=False)].copy()
    colors = ["#888888", "#4C9BE8", "#E87B4C", "#50C878", "#9B59B6"]
    ax = axes[0]
    bars = ax.bar(range(len(main)), main["accuracy"],
                  color=colors[: len(main)])
    ax.set_xticks(range(len(main)))
    ax.set_xticklabels(main["config_name"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("GSM8K Accuracy")
    ax.set_title("TALMAS Ablation Study — GSM8K")
    ax.set_ylim(0, max(main["accuracy"].max() * 1.15, 0.1))
    for bar, val in zip(bars, main["accuracy"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Plot 2: μ sweep for Full TALMAS
    mu_rows = df[df["config_name"].str.contains(r"μ=", regex=False)]
    ax2 = axes[1]
    if not mu_rows.empty:
        mu_vals  = mu_rows["mu"].tolist()
        acc_vals = mu_rows["accuracy"].tolist()
        ax2.plot(mu_vals, acc_vals, "o-", color="#9B59B6", linewidth=2, markersize=8)
        ax2.set_xlabel("μ (mask→mask suppression scale)")
        ax2.set_ylabel("GSM8K Accuracy")
        ax2.set_title("Full TALMAS: μ Sweep")
        ax2.set_xticks(mu_sweep_values)
    else:
        ax2.set_title("μ sweep — no data")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    is_instruct = "instruct" in args.model.lower()
    cfg = INSTRUCT_CONFIG if is_instruct else BASE_CONFIG

    if args.steps:
        cfg.steps = args.steps
    if args.generation_length:
        cfg.generation_length = args.generation_length

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model once with eager attention (required for TALMAS hook)
    tokenizer, model = load_model_and_tokenizer(args.model, eager_attn=True)
    mask_token_id, eos_token_id = resolve_special_tokens(tokenizer, model)
    print(f"mask_token_id={mask_token_id}, eos_token_id={eos_token_id}")
    print(f"Steps: {cfg.steps}  GenLen: {cfg.generation_length}  Samples: {args.max_samples}\n")

    # Load dataset once
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Running ablation on {len(dataset)} examples per config.\n")

    all_results = []

    for meta in ABLATION_CONFIGS:
        if meta["id"] == 5:
            # Sweep μ for Full TALMAS
            for mu in MU_SWEEP:
                swept_meta = dict(meta)
                swept_meta["name"] = f"Full TALMAS (μ={mu})"
                talmas_cfg = TALMASConfig(
                    lambda_max=meta["lambda_max"],
                    mu=mu,
                    use_timestep_gate=meta["use_timestep_gate"],
                    use_layer_gate=meta["use_layer_gate"],
                )
                result = run_one_config(
                    model=model, tokenizer=tokenizer, device=device,
                    mask_token_id=mask_token_id, eos_token_id=eos_token_id,
                    cfg=cfg, talmas_cfg=talmas_cfg, is_instruct=is_instruct,
                    dataset=dataset, config_meta=swept_meta,
                )
                all_results.append(result)
        else:
            talmas_cfg = TALMASConfig(
                lambda_max=meta["lambda_max"],
                mu=meta["mu"],
                use_timestep_gate=meta["use_timestep_gate"],
                use_layer_gate=meta["use_layer_gate"],
            )
            result = run_one_config(
                model=model, tokenizer=tokenizer, device=device,
                mask_token_id=mask_token_id, eos_token_id=eos_token_id,
                cfg=cfg, talmas_cfg=talmas_cfg, is_instruct=is_instruct,
                dataset=dataset, config_meta=meta,
            )
            all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"talmas_ablation_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df[["config_name", "lambda_max", "mu", "n_samples", "accuracy"]].to_string(index=False))

    # Plots
    plot_path = os.path.join(args.output_dir, f"talmas_ablation_{ts}.png")
    make_plots(df, MU_SWEEP, plot_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TALMAS ablation study on GSM8K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=100,
                        dest="max_samples",
                        help="Samples per config (None = full test set)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override diffusion steps (default: model preset)")
    parser.add_argument("--generation-length", type=int, default=None,
                        dest="generation_length",
                        help="Override generation length")
    parser.add_argument("--output-dir", type=str, default="results",
                        dest="output_dir",
                        help="Directory for CSV and plot outputs")
    main(parser.parse_args())
