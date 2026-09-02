#!/bin/bash
#SBATCH --job-name=download_models
# NOTE: the partition running this job must have internet access. On LEONARDO the
# GPU partition (boost_usr_prod) does not, so the serial partition is used here.
# If your cluster differs, switch the partition/qos below or run the srun bodies
# directly on a login node.
#SBATCH --partition=lrd_all_serial
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/dev/null


# TODO: Add path to the root dir of the GaMS-Translator repository
WORK_DIR=

# container
CONTAINER_DIR=$WORK_DIR/singularity
CONTAINER=$CONTAINER_DIR/gams_translator.sif

# download targets
MODELS_DIR=$WORK_DIR/models
REWARD_MODELS_DIR=$WORK_DIR/translation_reward_models
GRPO_DATA_DIR=$WORK_DIR/data/grpo_training

# HuggingFace cache. Kept inside WORK_DIR so the (small) home quota is not filled.
HF_CACHE_DIR=$WORK_DIR/.cache/huggingface

mkdir -p $MODELS_DIR
mkdir -p $REWARD_MODELS_DIR
mkdir -p $GRPO_DATA_DIR
mkdir -p $HF_CACHE_DIR

LOG_FILE=$WORK_DIR/download_logs.txt

# send everything below to the log file
echo "Logging to ${LOG_FILE}"
exec >> "$LOG_FILE" 2>&1

# TODO: Enter your HuggingFace access token.
# Required for Unbabel/wmt22-cometkiwi-da, which is a gated repository: you must
# also accept its license at https://huggingface.co/Unbabel/wmt22-cometkiwi-da
export HF_TOKEN=

# singularity forwards the host environment, so the container picks these up
export HF_HOME=/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

# mount points:
#   /models:        $MODELS_DIR
#   /reward_models: $REWARD_MODELS_DIR
#   /grpo_data:     $GRPO_DATA_DIR
#   /hf_cache:      $HF_CACHE_DIR
BINDS=$MODELS_DIR:/models,$REWARD_MODELS_DIR:/reward_models,$GRPO_DATA_DIR:/grpo_data,$HF_CACHE_DIR:/hf_cache

# On containers with huggingface_hub < 0.34 the CLI is called `huggingface-cli`
# instead of `hf`; replace `hf download` with `huggingface-cli download` below.

echo "===== 1/5 Downloading cjvt/GaMS3-12B ====="
srun \
  --cpu-bind=verbose \
  singularity exec \
    -B $BINDS \
    $CONTAINER \
    hf download cjvt/GaMS3-12B \
      --local-dir /models/GaMS3-12B

# Only the tokenizer files are needed from the Instruct model (it is used as
# TOKENIZER_PATH in sft/run_sft.sbatch for its chat template), so --include
# restricts the download to those and skips the model weights entirely.
echo "===== 2/5 Downloading cjvt/GaMS3-12B-Instruct tokenizer ====="
srun \
  --cpu-bind=verbose \
  singularity exec \
    -B $BINDS \
    $CONTAINER \
    hf download cjvt/GaMS3-12B-Instruct \
      --local-dir /models/GaMS3-12B-Instruct \
      --include "tokenizer*" "special_tokens_map.json" "added_tokens.json" "chat_template*"

echo "===== 3/5 Downloading Unbabel/wmt22-cometkiwi-da ====="
srun \
  --cpu-bind=verbose \
  singularity exec \
    -B $BINDS \
    $CONTAINER \
    hf download Unbabel/wmt22-cometkiwi-da \
      --local-dir /reward_models/wmt22-cometkiwi-da

echo "===== 4/5 Downloading fasttext LID model ====="
cd $REWARD_MODELS_DIR
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# The GRPO training data is a dataset repository (not a model), hence --repo-type dataset.
# The repo stores its parquet shards under data/, so they land in $GRPO_DATA_DIR/data/
# and grpo/run_grpo.sbatch binds that inner dir as /data.
echo "===== 5/5 Downloading cjvt/GaMS-Translator-GRPO-Training ====="
srun \
  --cpu-bind=verbose \
  singularity exec \
    -B $BINDS \
    $CONTAINER \
    hf download cjvt/GaMS-Translator-GRPO-Training \
      --repo-type dataset \
      --local-dir /grpo_data

echo "# completed at $(date)"
