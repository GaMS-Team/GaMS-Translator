#!/bin/bash
#SBATCH --job-name=merge_lora
#SBATCH --partition=boost_usr_prod
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/dev/null

# TODO: Add path to the root dir of the GaMS-Translator repository
WORK_DIR=

CONTAINER_DIR=$WORK_DIR/singularity
CONTAINER=$CONTAINER_DIR/gams_translator.sif
MODELS_DIR=$WORK_DIR/models

SCRIPT_DIR=$WORK_DIR/merge_lora

base_model_path=$1
tokenizer_path=$2
# Either sft, dpo/training or grpo
training_dir=$3
adapter_checkpoint=$4
output_name=$5

ADAPTER_PATH=$WORK_DIR/$training_dir/experiments/$adapter_checkpoint

mkdir -p $SCRIPT_DIR/logs

# run with apptainer
srun \
  --cpu-bind=verbose \
  --output=$SCRIPT_DIR/logs/"${output_name}.txt" \
  apptainer exec \
    --nv \
    -B $MODELS_DIR:/models,$ADAPTER_PATH:/adapter_input_path,$SCRIPT_DIR:/script \
    $CONTAINER \
    python3 /script/merge.py \
    --base_model=$base_model_path \
    --tokenizer=$tokenizer_path \
    --adapter=/adapter_input_path \
    --merged_model=/models/$output_name
