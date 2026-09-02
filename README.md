# GaMS-Translator

This repository contains the code for the training part of `Preference Optimization Pipeline and LLM-Based Machine Translator for English and Slovene Pairs` SLAIF service. It contains sample code for training an English → Slovene document-level translation model based on `cjvt/GaMS3-12B`. The model is trained in three consecutive stages — supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO) — each using LoRA. The code and environment are optimized for the **LEONARDO** EuroHPC supercomputer (SLURM + Singularity).

---

## Pipeline Overview

The stages must be run in order, because each one consumes the merged model produced by the previous one:

| # | Stage | Directory | Section |
|---|---|---|---|
| 1 | Build the Singularity container | `singularity/` | [1](#1-environment-setup) |
| 2 | Download base model, tokenizer, reward models and GRPO data | — | [2](#2-model-and-data-download) |
| 3 | SFT training → merge LoRA | `sft/` | [3](#3-sft-training), [4](#4-merging-lora-adapters) |
| 4 | Generate on-policy translations for DPO | `dpo/get_translations/` | [5](#5-generating-translations-for-dpo) |
| 5 | Build DPO preference pairs + COMET filtering | `dpo/prepare_data/` | [6](#6-dpo-data-preparation) |
| 6 | DPO training → merge LoRA | `dpo/training/` | [7](#7-dpo-training), [4](#4-merging-lora-adapters) |
| 7 | GRPO training → merge LoRA | `grpo/` | [8](#8-grpo-training), [4](#4-merging-lora-adapters) |

```
GaMS3-12B ──SFT──> merge ──> GaMS3-12B-SFT-Translator
                                      │
                                      ├──> generate translations ──> DPO pairs
                                      │
                                      └──DPO──> merge ──> GaMS3-12B-DPO-Translator
                                                                    │
                                                                    └──GRPO──> merge ──> GaMS3-12B-GRPO-Translator
```

### Repository Structure

```
GaMS-Translator/
├── singularity/
│   └── recipe.def              # Container definition
├── download_models.sbatch      # Downloads base model, tokenizer, reward models
├── data/
│   ├── sft_training/           # SFT train/validation data (ships with the repo)
│   ├── dpo_data/               # English source documents for DPO generation
│   └── grpo_training/          # GRPO data (downloaded in section 2)
├── sft/
│   ├── gams_sft.py
│   ├── deepspeed_config.json
│   └── run_sft.sbatch
├── dpo/
│   ├── get_translations/       # On-policy generation with vLLM
│   │   ├── translate.py
│   │   └── run_translation.sbatch
│   ├── prepare_data/           # Preference pair construction + COMET scoring
│   │   ├── prepare_wikipedia.py
│   │   ├── prepare_cc_news.py
│   │   ├── compute_comet.py
│   │   ├── select_training_data.py
│   │   └── run_preparation.sbatch
│   └── training/
│       ├── dpo_train.py
│       ├── deepspeed_config.json
│       └── run_dpo.sbatch
├── grpo/
│   ├── gams_grpo.py
│   ├── reward_functions.py
│   ├── deepspeed_config.json
│   └── run_grpo.sbatch
└── merge_lora/
    ├── merge.py
    └── run_merge.sbatch
```

### A Note on `WORK_DIR`

**Every** sbatch script in this repository starts with an empty `WORK_DIR` that you must fill in with the absolute path to your clone:

```bash
# TODO: Add path to the root dir of the GaMS-Translator repository
WORK_DIR=/leonardo_work/<your_account>/GaMS-Translator
```

All other paths (container, models, data, logs) are derived from it. Scripts that log to WandB additionally have a `WANDB_API_KEY` TODO. These TODOs are pointed out again in each section below.

---

## 1. Environment Setup

### Clone the Repository

```bash
git clone https://github.com/GaMS-Team/GaMS-Translator.git
cd GaMS-Translator
```

### What is Singularity?

Singularity (whose modern fork is called Apptainer) is a container runtime designed for HPC environments. Unlike Docker, it does not require root privileges at run time, which makes it suitable for shared SLURM clusters where users have no administrator access. It packages a full software environment — Python, CUDA libraries, pip packages — into a single `.sif` image file that runs reproducibly on any compatible node. LEONARDO provides `singularity`, which is the command used throughout this repository.

The container is based on `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` and installs the following into a virtual environment at `/opt/venv`:

| Library | Version |
|---|---|
| PyTorch | 2.8.0 |
| Transformers | 4.56.1 |
| TRL | 0.25.1 |
| DeepSpeed | 0.18.1 |
| vLLM | 0.10.2 |
| FlashAttention | 2.8.3 (prebuilt wheel for cu128 / torch 2.8) |
| WandB | 0.25.1 |
| PEFT, Accelerate | latest |
| Liger Kernel | latest |
| unbabel-comet, fasttext, sacrebleu, loguru | latest |

`/opt/venv/bin` is placed on `PATH` in the recipe's `%environment` section, so `python3` inside the container already resolves to the virtual environment. **No `source .../activate` call is needed in any job script.**

### Build the Image

The recipe is at `singularity/recipe.def`. Build the image into the same directory:

```bash
cd singularity/
singularity build gams_translator.sif recipe.def
cd ..
```

Building takes few minutes depending on network speed. The resulting `singularity/gams_translator.sif` is used by every SLURM script in this repository. You only need to build it once.

> **Note:** Building a container requires either root access or the `--fakeroot` flag on clusters that support it. If plain `singularity build` fails, try `singularity build --fakeroot`, or build on a machine where you have root and copy the `.sif` over. Check with your cluster administrators if neither works.

---

## 2. Model and Data Download

`download_models.sbatch` fetches everything the pipeline needs in one job:

| Repository | Destination | Purpose |
|---|---|---|
| `cjvt/GaMS3-12B` | `models/GaMS3-12B` | Base model for SFT |
| `cjvt/GaMS3-12B-Instruct` (tokenizer files only) | `models/GaMS3-12B-Instruct` | Tokenizer + chat template |
| `Unbabel/wmt22-cometkiwi-da` | `translation_reward_models/wmt22-cometkiwi-da` | COMET quality scoring (DPO filtering, GRPO reward) |
| fastText `lid.176.bin` | `translation_reward_models/lid.176.bin` | Language identification (GRPO reward) |
| [`cjvt/GaMS-Translator-GRPO-Training`](https://huggingface.co/datasets/cjvt/GaMS-Translator-GRPO-Training) | `data/grpo_training` | GRPO training + validation data |

Only the tokenizer files are pulled from the Instruct model, using `hf download --include "tokenizer*" "special_tokens_map.json" "added_tokens.json" "chat_template*"`. This fetches a few MB rather than the full ~24 GB of weights. The chat template is included because SFT uses the Instruct tokenizer specifically for it.

The GRPO data is a **dataset** repository rather than a model, so it is fetched with `--repo-type dataset`. It is public (no token needed) and licensed **CC BY-NC 4.0** — note the non-commercial clause, which is more restrictive than this repository's own Apache 2.0 code licence. The download is ~406 MB.

Because the dataset repo keeps its parquet shards in a `data/` subdirectory, they land at `data/grpo_training/data/`, and `grpo/run_grpo.sbatch` binds that inner directory as `/data`. See [section 8](#8-grpo-training) for the schema.

Only GRPO's data is downloaded. SFT data ships with the repository in `data/sft_training/`, and DPO's data is generated by the pipeline itself in sections 5 and 6.

### Before Running: Required Setup

Open `download_models.sbatch` and set:

```bash
WORK_DIR=/path/to/GaMS-Translator

# TODO: Enter your HuggingFace access token.
export HF_TOKEN=
```

The token is **required**: `Unbabel/wmt22-cometkiwi-da` is a gated repository. You must also accept its licence at [https://huggingface.co/Unbabel/wmt22-cometkiwi-da](https://huggingface.co/Unbabel/wmt22-cometkiwi-da) while logged in, otherwise the download fails with a 401.

### Partition Note

The job requests `--partition=lrd_all_serial` and no GPU. This is deliberate: on LEONARDO the GPU partition `boost_usr_prod` has **no internet access**, so a download job submitted there would fail. If your cluster differs, change the partition, or run the `srun` bodies directly on a login node.

`HF_HOME` is redirected to `$WORK_DIR/.cache/huggingface` so that ~25 GB of downloads do not fill your (small) home quota.

### Running

```bash
sbatch download_models.sbatch
```

Progress is written to `$WORK_DIR/download_logs.txt`. The job verifies that the COMET checkpoint landed at the exact path `compute_comet.py` expects and fails loudly if not.

> If your container has `huggingface_hub < 0.34`, the CLI is named `huggingface-cli` rather than `hf`. A comment in the script marks the lines to change.

---

## 3. SFT Training

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all weights of a large model, LoRA inserts small trainable matrices (adapters) into specific layers and freezes the original weights. This dramatically reduces trainable parameters and GPU memory while achieving quality close to full fine-tuning. All three training stages in this repository use LoRA with an identical configuration.

### Code Location

```
sft/
├── gams_sft.py           # Main training script
├── deepspeed_config.json # DeepSpeed ZeRO-3 configuration
└── run_sft.sbatch        # SLURM job submission script
```

### Input Data

SFT reads two JSONL files from `data/sft_training/`, mounted into the container as `/data`. Each line uses the standard TRL prompt/completion format, which keeps the model input separate from the expected output so that loss is computed only on completion tokens (`completion_only_loss=True`):

```json
{
  "corpus": "fineweb",
  "prompt":     [{"role": "user",      "content": "<translation instruction + English document>"}],
  "completion": [{"role": "assistant", "content": "<Slovene translation>"}]
}
```

### Before Running: Required Setup

Open `sft/run_sft.sbatch` and fill in:

```bash
WORK_DIR=/path/to/GaMS-Translator

# TODO: Enter your API key
export WANDB_API_KEY=
```

The base model and tokenizer default to the local copies created in section 2:

```bash
MODEL_INPUT_PATH=/models/GaMS3-12B
TOKENIZER_PATH=/models/GaMS3-12B-Instruct
```

### Submitting the Job

The script accepts four `--key=value` arguments:

```bash
sbatch sft/run_sft.sbatch \
    --lora_rank=64 \
    --warmup_steps=200 \
    --learning_rate=2e-5 \
    --min_lr=1e-6
```

The job requests 1 node, 4 GPUs, 32 CPUs, exclusive access, with an 8-hour limit.

### Key Configuration

**LoRA** (`use_lora` in `gams_sft.py`):

- `r` (rank): set by `--lora_rank`. Higher rank means more trainable parameters and potentially better quality, at the cost of memory and compute.
- `lora_alpha`: `2 * rank`, the scaling factor for LoRA updates.
- `lora_dropout`: 0.1.
- `target_modules`: all 7 projection layers — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Targeting all projections rather than just query and value typically improves quality.

**Data and batching:**

- `MAX_LENGTH`: 8192 tokens.
- `micro_batch_size`: 1 per GPU; global `batch_size`: 4. `gradient_accumulation_steps` is computed automatically as `batch_size / (world_size * micro_batch_size)`.

**Schedule:** 1 epoch, `cosine_with_min_lr` scheduler with warmup from `--warmup_steps` and a floor from `--min_lr`. Evaluation and checkpointing happen twice per epoch.

### Optimization Techniques

All three stages share the same `deepspeed_config.json` (byte-identical across `sft/`, `dpo/training/` and `grpo/`).

**DeepSpeed ZeRO Stage 3:** ZeRO (Zero Redundancy Optimizer) Stage 3 shards model parameters, gradients *and* optimizer states across all GPUs — unlike earlier stages, even the weights themselves are distributed, allowing models far larger than a single GPU's memory to be trained. Key settings:

- `bf16.enabled: true` — bfloat16 is more numerically stable than float16 for training.
- `stage3_gather_16bit_weights_on_model_save: true` — reassembles full weights when saving, so checkpoints are in standard format. **Required**, otherwise ZeRO-3 checkpoints are unusable.
- `overlap_comm: true` — overlaps gradient communication with computation.
- Batch sizes and gradient accumulation are `"auto"`, read from the training arguments at runtime.

**FlashAttention 2:** a memory-efficient, faster implementation of self-attention. Enabled via `attn_implementation="flash_attention_2"`, which matters a great deal at these sequence lengths.

**Gradient Checkpointing:** instead of keeping every intermediate activation in memory for the backward pass, activations are recomputed on the fly. This trades roughly 20–30% throughput for a large memory saving, and is what makes 12B + long documents fit.

**Liger Kernel:** provides fused, memory-efficient GPU kernels for common LLM operations (RMSNorm, RoPE, SwiGLU, cross-entropy). Enabled **for SFT only** via `use_liger_kernel=True`.

> **Why Liger is not used for DPO and GRPO.** In SFT, `use_liger_kernel` only patches operations *inside* the model's own forward pass, which stays compatible with DeepSpeed's just-in-time parameter gathering. In DPO and GRPO, TRL instead swaps in a fused-linear loss that reaches into the model, pulls `lm_head.weight` out directly and hands it to the kernel — with no `deepspeed.zero.GatheredParameters` guard. Under ZeRO-3 that weight is an unmaterialized shard at that moment, so training fails. This was confirmed experimentally, and `use_liger_kernel` is therefore deliberately absent from `DPOConfig` and `GRPOConfig`.

### WandB Logging

WandB is an experiment tracking platform that logs metrics (loss, learning rate, reward components) to a web dashboard. To use it:

1. Create a free account at [https://wandb.ai](https://wandb.ai).
2. Copy your API key from Settings → API Keys.
3. Set `WANDB_API_KEY` in the sbatch script.

All scripts set `WANDB_MODE=offline`, because LEONARDO's GPU partition has no internet access. Runs are written to the experiment directory and can be uploaded afterwards from a login node with `wandb sync`. If you do not want WandB at all, set `report_to="none"` in the training config.

### Resuming from a Checkpoint

Checkpoints are saved to `sft/experiments/<version>/checkpoint-<step>/`. To resume, uncomment and correct this line in the sbatch script:

```bash
# TODO: Uncomment and correct the checkpoint number if you are resuming from a checkpoint
# CKPT_PATH=$OUTPUT_DIR/checkpoint-42
```

While it stays commented, training starts from scratch. The same mechanism exists in the DPO and GRPO scripts.

---

## 4. Merging LoRA Adapters

This step runs **after every training stage**. SFT, DPO and GRPO all save LoRA adapters only; the next stage expects a complete standalone model.

### Code Location

```
merge_lora/
├── merge.py          # Merging script
└── run_merge.sbatch  # SLURM job submission script
```

### How Merging Works

During training LoRA keeps the base weights frozen and trains only the adapter matrices. At inference the adapter can either be applied on top of the frozen base model, or permanently merged into it. Merging adds the adapter contribution directly into the base parameters and discards the adapter structure, producing a standard checkpoint with no PEFT dependency. `merge.py` loads the base model in bfloat16, applies the adapter with `PeftModel.from_pretrained`, calls `merge_and_unload()`, and saves both model and tokenizer so the output directory is self-contained.

> Merging is done as a **separate job on purpose**. Merging inside the training script would run under DeepSpeed ZeRO-3, where parameters are sharded across ranks and every rank would race to write the same files — producing corrupt output. This single-GPU job sees complete weights.

### Before Running: Required Setup

Open `merge_lora/run_merge.sbatch` and set `WORK_DIR`.

### Running the Merge Job

The script takes five positional arguments:

1. `base_model_path` — the model the training stage *started from*
2. `tokenizer_path` — tokenizer for the merged model (may equal the base model path)
3. `training_dir` — repository subdirectory of the stage: `sft`, `dpo/training`, or `grpo`
4. `adapter_checkpoint` — checkpoint path relative to `<training_dir>/experiments/`
5. `output_name` — directory name for the merged model under `models/`

**After SFT:**

```bash
sbatch merge_lora/run_merge.sbatch \
    /models/GaMS3-12B \
    /models/GaMS3-12B-Instruct \
    sft \
    r-64_ws-200_lr-2e-5_min-lr-1e-6/checkpoint-1234 \
    GaMS3-12B-SFT-Translator
```

**After DPO:**

```bash
sbatch merge_lora/run_merge.sbatch \
    /models/GaMS3-12B-SFT-Translator \
    /models/GaMS3-12B-SFT-Translator \
    dpo/training \
    beta-0.1_r-64_ws-100_lr-5e-6_min-lr-1e-7/checkpoint-567 \
    GaMS3-12B-DPO-Translator
```

**After GRPO:** the same pattern, with `grpo` as `training_dir` and the DPO model as the base.

The merged model is saved to `models/<output_name>/` and logs to `merge_lora/logs/<output_name>.txt`. The job uses a single GPU, 64 GB RAM, 30-minute limit.

---

## 5. Generating Translations for DPO

### Why On-Policy Generation?

DPO learns from pairs of a **preferred** ("chosen") and **rejected** response to the same prompt. The rejected sample should reflect mistakes the model being trained actually makes — this is what *on-policy* means. So instead of using a generic dataset, the SFT model translates a set of English documents itself, and its own systematic errors become the rejected half of each pair.

For this corpus the SFT model makes three recurring formatting errors, which the preparation scripts in section 6 exploit:

1. Double newlines (paragraph separators) are collapsed to single newlines.
2. The model does not stop at the end of the document and keeps generating.
3. Dates are left in English format.

### Code Location

```
dpo/get_translations/
├── translate.py            # vLLM batch translation
└── run_translation.sbatch  # SLURM job submission script
```

### Input Data

Two source files ship with the repository, 1000 documents each:

```
data/dpo_data/
├── wikipedia_en.jsonl   # {"id": "wikipedia_0", "text": "# Title\n\n<document>"}
└── ccnews_en.jsonl      # {"id": "cc_news_0",   "text": "# Title\n*2020-06-30*\n\n<document>"}
```

### What the Script Does

`translate.py` loads the model once with vLLM, then translates the documents in batches using `model.chat()` with greedy decoding (`temperature=0`). Results are written **immediately after each batch**, so an interrupted job keeps its partial output. Each document produces a pair of files named by its position in the input:

```
data/dpo_data/<model>/<corpus>_translated/
├── 0_en.txt   # English source
├── 0_sl.txt   # Slovene translation
├── 1_en.txt
└── ...
```

Tensor parallelism is set automatically from `--tp_size=$SLURM_GPUS_ON_NODE` (4 GPUs by default).

### Before Running: Required Setup

Set `WORK_DIR` in `dpo/get_translations/run_translation.sbatch`.

### Running

The script takes three positional arguments — model name (a directory under `models/`), batch size, and corpus:

```bash
sbatch dpo/get_translations/run_translation.sbatch GaMS3-12B-SFT-Translator 32 wikipedia
sbatch dpo/get_translations/run_translation.sbatch GaMS3-12B-SFT-Translator 32 ccnews
```

Run it **once per corpus**; both are needed for section 6. The job uses 1 node with 4 GPUs and a 2-hour limit. Logs go to `dpo/get_translations/logs/<model>/<corpus>_translation.txt`.

> There is no resume logic — re-running restarts from index 0 and overwrites. With 1000 documents across 4 GPUs this is comfortably within the time limit.

**`translate.py` arguments:**

| Argument | Description |
|---|---|
| `--input_path` | JSONL file with one `{"id", "text"}` object per line |
| `--output_path` | Directory for the `<idx>_en.txt` / `<idx>_sl.txt` pairs |
| `--model` | Model directory or HuggingFace ID |
| `--tp_size` | Number of GPUs for tensor parallelism |
| `--batch_size` | Documents translated concurrently |
| `--max_seq_len` | Max output tokens (default 8192; context is set to `2 × max_seq_len`) |

---

## 6. DPO Data Preparation

### Code Location

```
dpo/prepare_data/
├── prepare_wikipedia.py     # Wikipedia preference pairs
├── prepare_cc_news.py       # CC-News preference pairs (also fixes dates)
├── compute_comet.py         # COMET quality scoring
├── select_training_data.py  # Threshold filtering + train/validation split
└── run_preparation.sbatch   # Runs all four steps in order
```

### The Four Steps

`run_preparation.sbatch` chains the whole pipeline. Only step 3 runs inside the container (COMET needs the GPU and the `comet` package); the other steps use nothing but the Python standard library and run directly on the node.

1. **`prepare_wikipedia.py`** — reads the `<idx>_en.txt` / `<idx>_sl.txt` pairs and builds one preference record per document. The raw translation becomes `rejected`; a repaired version becomes `chosen`, produced by restoring paragraph separators and truncating to the English paragraph count. Documents whose translation is *truncated* are dropped (no usable pair), and documents already formatted correctly are skipped (nothing to learn).
2. **`prepare_cc_news.py`** — the same, plus header handling: the title is forced to `# ...`, and the English `*YYYY-MM-DD*` date is converted to the Slovene `*D. M. YYYY*` format.
3. **`compute_comet.py`** — scores every `chosen` translation with `wmt22-cometkiwi-da` and adds a `comet_score` field. CometKiwi is *reference-free*: it estimates quality from the source and the translation alone.
4. **`select_training_data.py`** — drops records below the COMET threshold, wraps each side in chat format with the translation prompt, shuffles, and splits 90% train / 10% validation.

### Before Running: Required Setup

Set `WORK_DIR` in `dpo/prepare_data/run_preparation.sbatch`. The COMET checkpoint must exist at `translation_reward_models/wmt22-cometkiwi-da/checkpoints/model.ckpt` (section 2 puts it there).

### Running

```bash
sbatch dpo/prepare_data/run_preparation.sbatch --model=GaMS3-12B-SFT-Translator
```

Optional arguments override the COMET thresholds and the shuffling seed:

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model whose translations to process — same name used in section 5 |
| `--wikipedia_comet_threshold` | `0.65` | Minimum COMET score to keep a Wikipedia pair |
| `--ccnews_comet_threshold` | `0.7` | Minimum COMET score to keep a CC-News pair |
| `--seed` | `42` | Random seed for shuffling |

```bash
sbatch dpo/prepare_data/run_preparation.sbatch \
    --model=GaMS3-12B-SFT-Translator \
    --wikipedia_comet_threshold=0.70 \
    --ccnews_comet_threshold=0.75
```

The CC-News threshold is higher by default because average COMET score on this dataset is slightly higher.

The job requests 1 GPU with a 1-hour limit. Progress is logged to `dpo/prepare_data/logs/<model>_preparation.txt`, and `select_training_data.py` prints how many examples survived filtering — worth checking, since aggressive thresholds can leave very little data.

### Output

```
data/dpo_data/<model>/dpo_training/<model>/
├── wikipedia.jsonl         # step 1
├── ccnews.jsonl            # step 2
├── wikipedia_comet.jsonl   # step 3
├── ccnews_comet.jsonl      # step 3
├── training.jsonl          # step 4  <- DPO input
└── validation.jsonl        # step 4  <- DPO input
```

Each line of `training.jsonl` holds a full preference pair:

```json
{
  "id": "wikipedia_42",
  "chosen":   [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<repaired>"}],
  "rejected": [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<raw>"}],
  "comet_score": 0.83
}
```

---

## 7. DPO Training

### What is DPO?

Direct Preference Optimization trains a model directly on preference pairs, without the separate reward model that classical RLHF requires. For each pair it increases the relative log-probability of the chosen response over the rejected one, while a KL term keeps the policy from drifting too far from a reference model. The strength of that constraint is the `beta` parameter: **higher beta means less deviation** from the reference.

The reference model here costs no extra memory. `ref_model=None` combined with `peft_config` makes `DPOTrainer` use the base model *with the LoRA adapters disabled* as the implicit reference, so only one copy of the weights is ever resident.

### Code Location

```
dpo/training/
├── dpo_train.py
├── deepspeed_config.json
└── run_dpo.sbatch
```

### Before Running: Required Setup

Open `dpo/training/run_dpo.sbatch` and fill in:

```bash
WORK_DIR=/path/to/GaMS-Translator

# TODO: Set the name of the SFT-trained model (a dir inside $MODELS_DIR) that DPO starts from.
INPUT_MODEL_NAME=GaMS3-12B-SFT-Translator

# TODO: Point this at the dir with training.jsonl + validation.jsonl produced by dpo/prepare_data
DATA_DIR=$WORK_DIR/data/dpo_training

# TODO: Enter your API key
export WANDB_API_KEY=
```

`INPUT_MODEL_NAME` is the merged model from section 4, and `DATA_DIR` must point at the directory from section 6 that contains `training.jsonl` and `validation.jsonl`. The tokenizer is read from the merged model, which carries its own.

### Submitting the Job

```bash
sbatch dpo/training/run_dpo.sbatch \
    --beta=0.1 \
    --lora_rank=64 \
    --warmup_steps=100 \
    --learning_rate=5e-6 \
    --min_lr=1e-7
```

The job requests 2 nodes × 4 GPUs (8 total), 32 CPUs per task, exclusive access, 4-hour limit.

### Key Configuration

- `MAX_LENGTH`: 8192 tokens, covering prompt + completion together. Note that a DPO example holds a full document *and* its translation, so this budget is tighter than the same number is for SFT; pairs that exceed it are truncated according to `truncation_mode`.
- `micro_batch_size` 1, global `batch_size` 8, 3 epochs, evaluation and checkpointing 4× per epoch.
- Learning rate defaults are an order of magnitude below SFT (`5e-6` vs `2e-5`), which is standard for preference tuning: the model is being nudged, not taught a new task.
- Optimizations: DeepSpeed ZeRO-3, FlashAttention 2, gradient checkpointing, bf16. **No Liger kernel** — see the explanation in section 3.

After training, merge the adapter (section 4) to produce `GaMS3-12B-DPO-Translator`.

---

## 8. GRPO Training

### What is GRPO?

Group Relative Policy Optimization is an online reinforcement learning method. For each prompt the model generates a **group** of candidate completions, each is scored by one or more reward functions, and the advantage of each candidate is computed relative to the group's mean. The policy is then updated to favour above-average candidates. Unlike DPO, which learns from a fixed dataset of pairs, GRPO generates fresh samples from the current policy at every step — which is why it needs a live inference server.

### Code Location

```
grpo/
├── gams_grpo.py           # Main training script
├── reward_functions.py    # The four reward functions
├── deepspeed_config.json
└── run_grpo.sbatch
```

### Three-Node Topology

Generation dominates GRPO's cost, so it is offloaded to a dedicated vLLM server. The job requests **3 nodes** and assigns roles by SLURM rank:

| Rank | Role |
|---|---|
| 0, 1 | Trainers — 8 GPUs total running DeepSpeed ZeRO-3 |
| 2 | Inference server — `trl vllm-serve` with tensor parallelism across its 4 GPUs |

Trainers reach the server over HTTP at `http://<node2>:8000`, passed to the script as `--vllm_url`. TRL keeps the server's weights in sync as training progresses: for a LoRA model it merges the adapter, pushes the updated weights, then unmerges. When training finishes, the trainer calls `scancel` on its own job to release the inference node.

### Reward Functions

`reward_functions.py` defines four rewards, combined with the weights in `reward_weights`:

| Reward | Weight | What it measures |
|---|---|---|
| `comet_score` | 1.0 | CometKiwi quality of the translation given the English source (reference-free) |
| `language_score` | 0.2 | fastText probability that the output is Slovene — penalizes wrong-language output |
| `length_score` | 0.2 | Word-count ratio against the source; peaks when lengths match, 0 at ≥2× |
| `bleu_score` | 0.1 | sacreBLEU against the reference translation |

COMET dominates deliberately; the other three are guard rails against specific failure modes (drifting into another language, truncating, rambling). Both reward models are read from `/reward_models`, which the sbatch binds to `translation_reward_models/`.

### Before Running: Required Setup

Open `grpo/run_grpo.sbatch` and fill in:

```bash
WORK_DIR=/path/to/GaMS-Translator

# TODO: Set the name of the DPO-trained model (a dir inside $MODELS_DIR) that GRPO starts from.
INPUT_MODEL_NAME=GaMS3-12B-DPO-Translator

# TODO: Enter your API key
export WANDB_API_KEY=
```

`DATA_DIR` needs no change — it already points at the dataset downloaded in section 2:

```bash
DATA_DIR=$WORK_DIR/data/grpo_training/data
```

### Input Data

GRPO trains on [`cjvt/GaMS-Translator-GRPO-Training`](https://huggingface.co/datasets/cjvt/GaMS-Translator-GRPO-Training), fetched in section 2. `gams_grpo.py` reads the parquet shards directly:

```python
dataset = load_dataset("parquet", data_files={
    "training":   "/data/training-*.parquet",
    "validation": "/data/validation-*.parquet",
})
```

Globs are used rather than exact filenames so the dataset can be re-sharded upstream without changing the code. Note the split names are `training` and `validation`, not the usual `train`.

| Split | Examples (before filtering) |
|---|---|
| `training` | 103,351 |
| `validation` | 1,044 |

Some of these rows are filtered out at load time — see [Instruction Filtering](#instruction-filtering) below.

| Column | Type | Used for |
|---|---|---|
| `prompt` | list of `{role, content}` | The instruction + English document the model translates |
| `chosen` | list of `{role, content}` | Reference translation — the target for `bleu_score` |
| `comet_score` | float | COMET score of the reference, from dataset construction |
| `translator` | string | Which system produced the reference (`gams2`, `gams3`, `gemini`, `translate-gemma`) |

`comet_score` and `translator` are not consumed by training. They are passed through to the reward functions as keyword arguments (because `remove_unused_columns=False`) and absorbed by their `**kwargs`, so they are available if you want to weight or filter on them.

### Instruction Filtering

The dataset is not fully homogeneous: most rows use the long "profesionalni prevajalec" instruction that `reward_functions.PROMPT_TEMPLATE` defines, but a small number (all with `translator == "gemini"`, though not all gemini rows) use a shorter one — `Prevedi naslednje besedilo v slovenščino.`

This matters because `extract_src` recovers the English source by stripping `PROMPT_TEMPLATE` from the prompt, and `str.removeprefix` is a silent no-op when the prefix does not match. Such a row would keep the Slovene instruction line inside its "source text", skewing the COMET reward (wrong `src`) and the length reward (inflated source word count), with no error raised.

`gams_grpo.py` therefore drops these rows right after loading:

```python
raw_sizes = {split: len(split_dataset) for split, split_dataset in dataset.items()}
dataset = dataset.filter(has_expected_instruction)
```

`has_expected_instruction` imports `PROMPT_TEMPLATE` from `reward_functions` so the filter and the extraction can never disagree. On the validation split this removes **26 of 1,044** rows, leaving 1,018. Both counts are reported on rank 0 at startup:

```
Train data size: <kept> (dropped <n> with unexpected instruction)
Val data size: 1018 (dropped 26 with unexpected instruction)
```

Watch those numbers on the first run — a large drop would mean the dataset's instruction format has diverged from `PROMPT_TEMPLATE`, which is worth investigating before spending GPU hours.

### Submitting the Job

```bash
sbatch grpo/run_grpo.sbatch \
    --beta=0.04 \
    --lora_rank=64 \
    --warmup_steps=100 \
    --learning_rate=2e-5 \
    --min_lr=1e-6
```

The job requests 3 nodes × 4 GPUs, `--qos=boost_qos_lprod`, and a 4-day limit (`--time=4-00`) — GRPO is by far the longest stage, because every step waits on generation.

### Key Configuration

- `MAX_PROMPT_LENGTH` / `MAX_COMPLETION_LENGTH`: 4096 each.
- `num_generations`: 4 completions sampled per prompt — the group size that advantages are computed against.
- `batch_size`: 64 prompts globally; `generation_batch_size`: 64 completions per vLLM call.
- `temperature`: 0.1. Low for a translation task, where the goal is a single correct output rather than diversity.
- `beta` defaults to 0.04, much lower than DPO's 0.1 — GRPO tolerates more drift from the reference.
- 3 epochs, evaluation and checkpointing 4× per epoch, `log_completions=True` so sample outputs appear in WandB.
- Optimizations: DeepSpeed ZeRO-3, FlashAttention 2, gradient checkpointing, bf16. **No Liger kernel** — see section 3.

> `eval_on_start=True` triggers a full generation pass over the validation set before the first training step. It is a useful baseline, but for GRPO it is not free — remove the flag if you would rather skip it.

After training, merge the adapter (section 4) to produce the final `GaMS3-12B-GRPO-Translator`.

---

## Acknowledgments

The service was developed at the **University of Ljubljana, Faculty of Computer and Information Science**.

The project was also supported by:

* **ARIS** (Slovenian Research and Innovation Agency).
* **NextGenerationEU**.
* European Union under Horizon Europe (101186647 – **AI4DH**)
* **EuroHPC JU**.
* **SLING** (Slovenian National Supercomputing Network).
* **NVIDIA** Academic Grant Program using DGX Spark.

---

## Contact

**Domen Vreš**  
domen.vres@fri.uni-lj.si

---

## License

This project is licensed under the **Apache 2.0** license.
