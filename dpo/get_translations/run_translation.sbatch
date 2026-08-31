#!/bin/bash
#SBATCH --job-name=dpo_translation
#SBATCH --output=translation_log.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=boost_usr_prod

# TODO: Add path to the root dir of the GaMS-Translator repository
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/dpo/get_translations

CONTAINER_DIR=$WORK_DIR/singularity
CONTAINER=$CONTAINER_DIR/gams_translator.sif

DATA_DIR=$WORK_DIR/data/dpo_data
MODELS_DIR=$WORK_DIR/models

model=$1
batch_size=$2
# Either wikipedia or ccnews
corpus=$3

OUTPUT_DIR=$DATA_DIR/${model}/${corpus}_translated
mkdir -p $OUTPUT_DIR

LOG_DIR=$SCRIPT_DIR/logs/${model}
mkdir -p $LOG_DIR
LOG_FILE=$LOG_DIR/${corpus}_translation.txt

srun \
    --cpu-bind=verbose \
    --output=$LOG_FILE \
    singularity exec \
        --nv \
        --bind $MODELS_DIR:/models,$DATA_DIR:/data,$SCRIPT_DIR:/script,$OUTPUT_DIR:/output \
        --pwd /script \
        $CONTAINER \
        python3 translate.py \
            --input_path=/data/${corpus}_en.jsonl \
            --output_path=/output \
            --model=/models/$model \
            --tp_size=${SLURM_GPUS_ON_NODE} \
            --batch_size=$batch_size
