"""
LLaDA GSM8K Benchmark Replication
===================================
Replicates the pure diffusion + low-confidence remasking results from the LLaDA paper.

Paper hyperparameters (GSM8K):
  - Base model:    generation_length=1024, steps=1024, 4-shot, low-confidence remasking
  - Instruct model: generation_length=512,  steps=512,  4-shot, low-confidence remasking
                    + EOS confidence forced to 0 to prevent early termination

Usage:
  # Base model
  python llada_gsm8k_eval.py --model GSAI-ML/LLaDA-8B-Base --split test --max_samples 100

  # Instruct model
  python llada_gsm8k_eval.py --model GSAI-ML/LLaDA-8B-Instruct --split test --max_samples 100

Requirements:
  pip install torch transformers datasets accelerate tqdm
"""

import re
import math
import argparse
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
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
# 4-shot GSM8K prompt  (same examples the paper would have used)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """\
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

"""


def build_prompt(question: str, is_instruct: bool) -> str:
    """Build the 4-shot prompt for a given question."""
    if is_instruct:
        # Chat-style prompt for the instruct model
        system = "Solve the following math problem step by step. At the end, state 'The answer is X.' where X is the numeric answer."
        shots = FEW_SHOT_EXAMPLES.strip()
        return (
            f"<|system|>\n{system}\n"
            f"<|user|>\n{shots}\n\nQuestion: {question}\nAnswer:"
            f"<|assistant|>\n"
        )
    else:
        # Plain few-shot for base model
        return FEW_SHOT_EXAMPLES + f"Question: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Core diffusion sampling  (Algorithm 5 from the paper — low-confidence remasking)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def low_confidence_remasking_sample(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,          # (1, prompt_len)  — already on device
    cfg: SamplingConfig,
    device: torch.device,
    mask_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """
    Pure diffusion sampling with low-confidence remasking (Algorithm 5).

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
        pred_ids   = torch.where(already_unmasked, current_response, pred_ids)

        # Number of tokens that should be unmasked at time s
        n_unmask = math.floor(L * (1.0 - s))
        n_unmask = max(0, min(n_unmask, L))

        # Select the n_unmask positions with the highest confidence
        _, top_indices = torch.topk(confidence, k=n_unmask, largest=True)
        new_response = torch.full((L,), mask_token_id, dtype=torch.long, device=device)
        new_response[top_indices] = pred_ids[top_indices]

        input_ids[0, prompt_len:] = new_response

    # Final response (drop everything after the first EOS if present)
    final = input_ids[0, prompt_len:]
    eos_positions = (final == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        final = final[: eos_positions[0]]

    return final


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer.
    Looks for 'The answer is X' or '#### X' patterns (GSM8K convention).
    """
    # GSM8K ground-truth format: #### <number>
    m = re.search(r"####\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # LLaDA generation format: "The answer is X"
    m = re.search(r"[Tt]he answer is\s*([\d,\-\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()

    # Last number in the string as fallback
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def answers_match(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return float(pred) == float(gold.replace(",", ""))
    except ValueError:
        return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    is_instruct = "instruct" in args.model.lower()
    cfg = INSTRUCT_CONFIG if is_instruct else BASE_CONFIG

    # Allow CLI overrides
    if args.generation_length:
        cfg.generation_length = args.generation_length
    if args.steps:
        cfg.steps = args.steps

    print(f"Model:             {args.model}")
    print(f"Mode:              {'Instruct' if is_instruct else 'Base'}")
    print(f"Generation length: {cfg.generation_length}")
    print(f"Sampling steps:    {cfg.steps}")
    print(f"Zero EOS conf:     {cfg.zero_eos_confidence}")
    print(f"Samples:           {args.max_samples or 'all'}")
    print()

    # ------------------------------------------------------------------ #
    # Load model and tokenizer                                             #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model...")
    model = AutoModel.from_pretrained(  # LLaDA custom arch — not in AutoModelForMaskedLM
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # spread across available GPUs
    )
    model.eval()

    # Identify special token IDs
    # LLaDA uses a custom tokenizer that does NOT set the standard HuggingFace
    # mask_token_id attribute. Instead its mask token is a regular vocab entry.
    # Resolution order:
    #   1. tokenizer.mask_token_id  (standard HF attribute — usually None for LLaDA)
    #   2. convert "[MASK]" string  (some tokenizer configs register it this way)
    #   3. model.config.mask_token_id  (custom config field LLaDA sets)
    #   4. hardcoded fallback 126336  (known value from GSAI-ML/LLaDA-8B-* repos)
    mask_token_id = tokenizer.mask_token_id

    if mask_token_id is None:
        # Try converting the string token
        encoded = tokenizer.convert_tokens_to_ids("[MASK]")
        if encoded != tokenizer.unk_token_id:
            mask_token_id = encoded

    if mask_token_id is None and hasattr(model, "config"):
        mask_token_id = getattr(model.config, "mask_token_id", None)

    if mask_token_id is None:
        # Hardcoded fallback — verified from the official GSAI-ML repo
        mask_token_id = 126336
        print(f"WARNING: mask_token_id not found in tokenizer/config, "
              f"falling back to known LLaDA value: {mask_token_id}")

    # EOS token — try standard attr, then config, then known fallback
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and hasattr(model, "config"):
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = 126081   # known LLaDA EOS token id
        print(f"WARNING: eos_token_id not found, falling back to: {eos_token_id}")

    print(f"mask_token_id={mask_token_id}, eos_token_id={eos_token_id}")

    # ------------------------------------------------------------------ #
    # Load GSM8K dataset                                                   #
    # ------------------------------------------------------------------ #
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} examples...\n")

    # ------------------------------------------------------------------ #
    # Eval loop                                                            #
    # ------------------------------------------------------------------ #
    correct = 0
    total   = 0
    results = []

    for example in tqdm(dataset, desc="GSM8K"):
        question  = example["question"]
        gold_full = example["answer"]          # contains #### <answer>
        gold_ans  = extract_answer(gold_full)  # just the number

        prompt_text = build_prompt(question, is_instruct)
        prompt_ids  = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(device)

        # Run pure-diffusion sampling
        output_ids = low_confidence_remasking_sample(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            cfg=cfg,
            device=device,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
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

    # ------------------------------------------------------------------ #
    # Report                                                               #
    # ------------------------------------------------------------------ #
    accuracy = correct / total * 100
    print(f"\n{'='*50}")
    print(f"GSM8K Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*50}")
    print(f"\nPaper reports: 70.3% (Base, 4-shot) / 69.4% (Instruct)")

    # Save results
    if args.output_file:
        import json
        with open(args.output_file, "w") as f:
            json.dump({
                "model":      args.model,
                "accuracy":   accuracy,
                "correct":    correct,
                "total":      total,
                "config":     cfg.__dict__,
                "results":    results,
            }, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    return accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replicate LLaDA GSM8K benchmark")

    parser.add_argument(
        "--model",
        type=str,
        default="GSAI-ML/LLaDA-8B-Base",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8K split to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of examples (None = full test set of 1319)",
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=None,
        help="Override generation length (default: 1024 for base, 512 for instruct)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override number of diffusion steps (default: same as generation_length)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="gsm8k_results.json",
        help="JSON file to save per-example results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-example predictions",
    )

    args = parser.parse_args()
    evaluate(args)
