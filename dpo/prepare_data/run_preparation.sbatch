#!/bin/bash
#SBATCH --job-name=prepare_dpo_data
#SBATCH --partition=boost_usr_prod
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=dpo_preparation.txt

# Full DPO data preparation pipeline. Consumes the output of dpo/get_translations
# and produces training.jsonl / validation.jsonl:
#
#   1. prepare_wikipedia.py     wikipedia translations -> wikipedia.jsonl
#   2. prepare_cc_news.py       ccnews translations    -> ccnews.jsonl
#   3. compute_comet.py         COMET score both files (needs GPU + container)
#   4. select_training_data.py  filter by COMET score, split into train/validation
#
# Only step 3 runs inside singularity; the other steps use nothing but the Python
# standard library and run directly on the node.
#
# Usage:
#   sbatch run_preparation.sbatch --model=<model_name> \
#     [--wikipedia_comet_threshold=0.65] [--ccnews_comet_threshold=0.7] [--seed=42]
#
# where <model_name> is the same model name that was passed to run_translation.sbatch.

set -e

# convert the --key=value arguments to variables
for argument in "$@"
do
  if [[ $argument == *"="* ]]; then
    key=$(echo $argument | cut -f1 -d=)
    value=$(echo $argument | cut -f2 -d=)
    if [[ $key == *"--"* ]]; then
        v="${key/--/}"
        declare ${v,,}="${value}"
   fi
  fi
done

if [ -z "${model:-}" ]; then
  echo "ERROR: --model=<model_name> is required (the model used in run_translation.sbatch)."
  exit 1
fi

# COMET thresholds default to the defaults of select_training_data.py
wikipedia_comet_threshold=${wikipedia_comet_threshold:-0.65}
ccnews_comet_threshold=${ccnews_comet_threshold:-0.7}
seed=${seed:-42}

# TODO: Add path to the root dir of the GaMS-Translator repository
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/dpo/prepare_data

# container
CONTAINER_DIR=$WORK_DIR/singularity
CONTAINER=$CONTAINER_DIR/gams_translator.sif

# data
DATA_DIR=$WORK_DIR/data/dpo_data/${model}

# COMET checkpoint is expected at $REWARD_MODELS_DIR/wmt22-cometkiwi-da/checkpoints/model.ckpt
REWARD_MODELS_DIR=$WORK_DIR/translation_reward_models

# translations produced by dpo/get_translations (paired <idx>_en.txt / <idx>_sl.txt files)
WIKIPEDIA_TRANSLATIONS=$DATA_DIR/wikipedia_translated
CCNEWS_TRANSLATIONS=$DATA_DIR/ccnews_translated

# all intermediate and final files of this run live here
OUTPUT_DIR=$DATA_DIR/dpo_training/${model}
mkdir -p $OUTPUT_DIR

LOG_DIR=$SCRIPT_DIR/logs
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/${model}_preparation.txt

# send everything below to the per-model log file
echo "Logging to ${LOG_FILE}"
exec >> "$LOG_FILE" 2>&1

echo "===== 1/4 Preparing Wikipedia DPO pairs ====="
python3 $SCRIPT_DIR/prepare_wikipedia.py \
  --input_dir=$WIKIPEDIA_TRANSLATIONS \
  --output_path=$OUTPUT_DIR/wikipedia.jsonl

echo "===== 2/4 Preparing CC News DPO pairs ====="
python3 $SCRIPT_DIR/prepare_cc_news.py \
  --input_dir=$CCNEWS_TRANSLATIONS \
  --output_path=$OUTPUT_DIR/ccnews.jsonl

# mount points for the COMET step:
#   /script:        $SCRIPT_DIR
#   /data:          $OUTPUT_DIR
#   /reward_models: $REWARD_MODELS_DIR
echo "===== 3/4 Computing COMET scores ====="
for corpus in wikipedia ccnews; do
  echo "--- COMET: ${corpus} ---"
  srun \
    --cpu-bind=verbose \
    singularity exec \
      --nv \
      -B ${SCRIPT_DIR}:/script,${OUTPUT_DIR}:/data,${REWARD_MODELS_DIR}:/reward_models \
      ${CONTAINER} \
      python3 /script/compute_comet.py \
        --input_file=/data/${corpus}.jsonl \
        --output_path=/data/${corpus}_comet.jsonl
done

echo "===== 4/4 Selecting training data ====="
python3 $SCRIPT_DIR/select_training_data.py \
  --wikipedia_input=$OUTPUT_DIR/wikipedia_comet.jsonl \
  --ccnews_input=$OUTPUT_DIR/ccnews_comet.jsonl \
  --output_dir=$OUTPUT_DIR \
  --wikipedia_comet_threshold=${wikipedia_comet_threshold} \
  --ccnews_comet_threshold=${ccnews_comet_threshold} \
  --seed=${seed}

echo "# completed at $(date)"
