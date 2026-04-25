# TALMAS on LLaDA-8B

**Timestep-Adaptive, Layer-Dependent Masked Attention Suppression** — a training-free inference intervention on a frozen LLaDA-8B checkpoint.

TALMAS applies a learned-free logit bias to the attention mechanism that suppresses attention to `[MASK]` tokens. The bias is gated by how many tokens have been revealed so far (timestep gate) and which layer we are in (layer gate), so suppression is heaviest where it matters most: deep layers, late in denoising.

---

## How It Works

During LLaDA's reverse diffusion process, every `[MASK]` token attends to other `[MASK]` tokens — positions that carry no real information yet. TALMAS adds a negative bias to those attention logits before the softmax:

```
â_ij = (Q_i · K_j^T) / sqrt(d_k)  −  λ(ℓ, t) · m_j · [(1 − m_i) + μ · m_i]
```

where `m_i = 1` if token `i` is `[MASK]`, else `0`.

| Query \ Key | Real | `[MASK]` |
|-------------|------|----------|
| Real        | no bias | `−λ` (full suppression) |
| `[MASK]`    | no bias | `−λ·μ` (partial; preserves coordination) |

The suppression magnitude `λ(ℓ, t)` is controlled by two gates:

```
λ(ℓ, t) = λ_max · f(1 − r_t) · g(ℓ / L)

f(x) = x²                        # timestep gate  — grows as more tokens are revealed
g(u) = sigmoid(8 · (u − 0.5))    # layer gate     — peaks at deep layers
```

At early denoising steps (`r_t ≈ 1`, almost everything masked): `f ≈ 0`, so `λ ≈ 0` — no interference.  
At late steps in deep layers (`r_t ≈ 0`, nearly done): `f ≈ 1`, `g ≈ 1`, so `λ ≈ λ_max` — maximum suppression.

No weights are modified. The bias is injected via a monkey-patch on `torch.nn.functional.scaled_dot_product_attention` and is removed after generation.

---

## Project Structure

```
llada/
├── src/
│   ├── config.py        # SamplingConfig, TALMASConfig, ablation presets
│   ├── sampling.py      # LLaDA Algorithm 5 (low-confidence remasking)
│   ├── talmas.py        # Gate functions + TALMASHookManager
│   └── utils.py         # Prompting, answer extraction, model loading
├── scripts/
│   ├── gsm8k_eval.py    # Unified evaluation CLI (baseline + TALMAS)
│   └── run_ablation.py  # Ablation study runner (5 configs + μ sweep)
├── results/             # JSON, CSV, and plot outputs
├── llada_gsm8k_eval.py  # Original baseline script (preserved, not modified)
├── talmas_colab.ipynb   # Colab notebook version
└── CLAUDE.md            # Full implementation spec
```

---

## Setup

```bash
pip install -q "transformers>=4.38.0,<4.44.0" accelerate datasets evaluate tqdm pandas matplotlib
```

Tested on Python 3.10+, A100 40 GB (Google Colab Pro). The model requires roughly 20 GB VRAM in bfloat16.

---

## Usage

### Baseline evaluation (replicates LLaDA paper)

```bash
# Base model  — paper target: 70.3%
python scripts/gsm8k_eval.py \
    --model GSAI-ML/LLaDA-8B-Base \
    --max_samples 100

# Instruct model  — paper target: 69.4%
python scripts/gsm8k_eval.py \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --max_samples 100
```

### TALMAS evaluation

```bash
python scripts/gsm8k_eval.py \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --talmas \
    --lambda-max 4.0 \
    --mu 0.1 \
    --max_samples 100 \
    --verbose
```

### Hyperparameter sweep

28 pre-defined configs enumerate a Tier 1 λ_max × μ grid (IDs 1–25) and a Tier 3 sigmoid slope sweep (IDs 26–28). Each config ID runs independently — one per machine.

**List all configs:**
```bash
python scripts/run_sweep.py --list-configs
```

**Run one config (one machine):**
```bash
python scripts/run_sweep.py --config-id 12 \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --max-samples 100 --steps 128 \
    --output-dir results/sweep
```

**Distribute across machines (example: 5 VMs, one λ row each):**
```bash
# VM 1: ids 1–5  (λ=1.0)
for id in 1 2 3 4 5; do
    python scripts/run_sweep.py --config-id $id \
        --max-samples 100 --steps 128 --output-dir results/sweep
done

# VM 2: ids 6–10  (λ=2.0), etc.
```

Each machine writes `results/sweep/sweep_cfg{id:02d}.jsonl` (one line per example) and appends one row to `results/sweep/sweep_results.csv`. Files from all machines can be merged by concatenating the CSVs. Checkpointing works per config ID: restart the same command after preemption and it resumes from where it left off.

**Tier 3 slope sweep:** after Tier 1 completes, update `lambda_max` and `mu` in `SWEEP_CONFIGS` ids 26–28 in `src/config.py` to the best values found, then run `--config-id 26/27/28`.

| Output file | Description |
|-------------|-------------|
| `results/sweep/sweep_cfg{id:02d}.jsonl` | Per-example results for config `id` |
| `results/sweep/sweep_results.csv` | Accumulated rows: config_id, λ_max, μ, slope, accuracy |

### Ablation study (all 5 configs + μ sweep)

```bash
python scripts/run_ablation.py \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --max-samples 100 \
    --output-dir results
```

Outputs `results/talmas_ablation_<timestamp>.csv` and `results/talmas_ablation_<timestamp>.png`.

### Checkpointing (resumable runs)

Pass `--checkpoint <file>` to write each result to a JSONL file immediately after it is evaluated. If the process is interrupted (e.g. a GCP spot instance preemption), rerun the **exact same command** and it will skip already-evaluated examples and continue from where it left off.

```bash
python scripts/gsm8k_eval.py \
    --model GSAI-ML/LLaDA-8B-Base \
    --max_samples 100 \
    --checkpoint results/ckpt_baseline.jsonl \
    --output-dir results
```

On restart after preemption:
```
Resuming from checkpoint 'results/ckpt_baseline.jsonl': 47 examples already done (33/47 correct).
Evaluating on 53 remaining examples...
```

The checkpoint file is append-only, so a mid-write crash cannot corrupt previously saved lines. The final JSON summary (`--output-dir`) is still written at the end as usual.

For extra safety on GCP, sync the checkpoint to Cloud Storage periodically:
```bash
gsutil cp results/ckpt_baseline.jsonl gs://your-bucket/talmas/
```

### Quick smoke-test

```bash
python scripts/run_ablation.py --max-samples 5 --steps 20
```

Config 1 (`λ_max=0`) should match the baseline exactly. If configs 2–5 all return the same accuracy as config 1, the hook is not firing — see Troubleshooting below.

---

## CLI Reference

### `scripts/gsm8k_eval.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `GSAI-ML/LLaDA-8B-Base` | HuggingFace model ID or local path |
| `--split` | `test` | GSM8K split (`train` or `test`) |
| `--max_samples` | all | Limit evaluation examples |
| `--generation_length` | 1024 / 512 | Override response token budget |
| `--steps` | 1024 / 512 | Override diffusion steps |
| `--output-dir` | `results` | Directory for auto-named JSON output |
| `--output_file` | — | Explicit output path |
| `--checkpoint` | — | JSONL file for incremental checkpointing; resumes if file exists |
| `--verbose` | off | Print per-example predictions |
| `--talmas` | off | Enable TALMAS suppression |
| `--lambda-max` | `4.0` | Maximum bias magnitude |
| `--mu` | `0.1` | `[MASK]→[MASK]` suppression scale |
| `--no-timestep-gate` | off | Disable `f(1−r_t)` gate |
| `--no-layer-gate` | off | Disable `g(ℓ/L)` gate |

Defaults for `generation_length` and `steps` are `1024` for base models and `512` for instruct models, matching the LLaDA paper hyperparameters.

### `scripts/run_ablation.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `GSAI-ML/LLaDA-8B-Instruct` | HuggingFace model ID |
| `--max-samples` | `100` | Samples per config |
| `--steps` | model preset | Override diffusion steps |
| `--generation-length` | model preset | Override generation length |
| `--output-dir` | `results` | CSV and plot output directory |

### `scripts/run_sweep.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--list-configs` | — | Print all 28 sweep configs and exit |
| `--config-id` | required | Sweep config ID to run (1–28) |
| `--model` | `GSAI-ML/LLaDA-8B-Instruct` | HuggingFace model ID |
| `--split` | `test` | GSM8K split |
| `--max-samples` | `100` | Limit evaluation examples |
| `--steps` | model preset | Override diffusion steps |
| `--generation-length` | model preset | Override generation length |
| `--output-dir` | `results/sweep` | Directory for JSONL checkpoints and CSV |

---

## Ablation Configurations

| ID | Name | `λ_max` | Timestep gate | Layer gate | `μ` |
|----|------|---------|---------------|------------|-----|
| 1 | Baseline (LLaDA) | 0.0 | — | — | 0.0 |
| 2 | Static Bias | 4.0 | ✗ | ✗ | 0.0 |
| 3 | Timestep-Only | 4.0 | ✓ | ✗ | 0.0 |
| 4 | Layer-Only | 4.0 | ✗ | ✓ | 0.0 |
| 5 | Full TALMAS | 4.0 | ✓ | ✓ | 0.1* |

\* Config 5 is also swept over `μ ∈ {0.0, 0.2, 0.5, 1.0}` in `run_ablation.py`.

---

## Implementation Notes

**Flash Attention compatibility.** TALMAS patches `torch.nn.functional.scaled_dot_product_attention`. Flash Attention 2 bypasses this path. When `--talmas` is active, the model is loaded with `attn_implementation="eager"` to guarantee the hook fires.

**r_t mapping.** LLaDA's timestep schedule is `linspace(1.0, 1/N, N)`. The value `t` at each step is the current mask ratio `r_t` passed directly to `compute_lambda`. No re-indexing needed.

**No gradient computation.** All inference runs under `@torch.inference_mode()`. The TALMAS bias is constructed with plain arithmetic (no autograd) and is safe under inference mode.

**Hook cleanup.** `TALMASHookManager.remove()` is called inside a `try/finally` block so original attention methods are always restored, even if generation raises an exception.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| All TALMAS configs identical to baseline | Hook not firing — FA2 in use | Confirm `attn_implementation="eager"` is set; run with `--talmas --lambda-max 100 --max_samples 1 --verbose` and check output changes |
| `MASK_TOKEN_ID is None` | Non-standard tokenizer | The fallback to `126336` should handle this; check the warning printed at startup |
| CUDA OOM | `generation_length` or `steps` too large | Reduce `--generation_length` and `--steps`; bfloat16 is already used |
| Accuracy identical across μ values | μ sweep not creating new hook managers | Each config in `run_ablation.py` creates a fresh `TALMASHookManager`; check for exception output |
| `logits` shape error | LLaDA output object changed | Access via `model(...).logits`; inspect return type with a single forward pass |
| Spot VM preempted, progress lost | No checkpoint file specified | Always pass `--checkpoint results/ckpt.jsonl`; restart with the same command to resume |
