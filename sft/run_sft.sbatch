#!/bin/bash
#SBATCH --job-name=translator_sft
#SBATCH --partition=boost_usr_prod
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/dev/null

# get the name of this script
if [ -n "${SLURM_JOB_ID:-}" ] ; then
  SBATCH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
  SBATCH=$(realpath "$0")
fi

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

# time of running script
DATETIME=`date "+%Y%m%d-%H%M"`
version=r-${lora_rank}_ws-${warmup_steps}_lr-${learning_rate}_min-lr-${min_lr}

# TODO: Add path to the root dir of the GaMS-PEFT-Tuning repository
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/sft

# Model path and tokenizer paths can also be local paths (in that case you should mount them into the container)
MODEL_INPUT_PATH=cjvt/GaMS3-12B
TOKENIZER_PATH=cjvt/GaMS3-12B-Instruct

# data
DATA_DIR=$WORK_DIR/data/sft_training

# container
CONTAINER_DIR=$WORK_DIR/singularity
CONTAINER_FILENAME=gams_translator.sif

# set experiment output dir
EXPERIMENTS_DIR=$SCRIPT_DIR/experiments
mkdir -p ${EXPERIMENTS_DIR}/${version}
WANDB_RUN_NAME=GaMS3_12B_${version}

OUTPUT_DIR=${EXPERIMENTS_DIR}/${version}

# TODO: Uncomment and correct the checkpoint number if you are resuming from a checkpoint
# CKPT_PATH=$OUTPUT_DIR/checkpoint-42
RESUME_ARG=""
if [ -n "$CKPT_PATH" ]; then
  echo "Resuming from checkpoint: $CKPT_PATH"
  RESUME_ARG="--resume_from_checkpoint $CKPT_PATH"
fi

# set run name
if [ "${version}" == "${DATETIME}" ]; then
  RUN_NAME=${version}
else
  RUN_NAME=${version}_R${DATETIME}
fi

# backup this script
cp -rp ${SBATCH} ${EXPERIMENTS_DIR}/${RUN_NAME}.sbatch

# execution script the script
SCRIPT=${EXPERIMENTS_DIR}/${RUN_NAME}.sh
touch $SCRIPT
chmod a+x $SCRIPT

IS_DISTRIBUTED=$([ 1 -lt $SLURM_JOB_NUM_NODES ] && echo " distributed over $SLURM_JOB_NUM_NODES nodes" || echo " on 1 node")


# setup the command
echo """#!/bin/bash

# using '$(basename $SBATCH)' -> $RUN_NAME.sbatch, running $SLURM_NPROCS tasks$IS_DISTRIBUTED

# ------------------------------------------------------------
# mount points:
# ------------------------------------------------------------
# /model_input_path: '$MODEL_INPUT_PATH'
# /tokenizer_path: '$TOKENIZER_PATH'
# /script: '$SCRIPT_DIR'
# /experiments: '$EXPERIMENTS_DIR'
# /data: '$DATA_DIR'
# /output: '$OUTPUT_DIR'
# ------------------------------------------------------------

# print info
echo -e \"\"\"
# starting at \$(date)
# running process \$SLURM_PROCID on \$SLURMD_NODENAME
\"\"\"

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export PYTHONUNBUFFERED=TRUE

# TODO: Enter your API key
export WANDB_API_KEY=
export WANDB_PROJECT=GaMS-Translator-SFT
export WANDB_DIR=/experiments
# Enable offline mode if the GPU partition does not have the access to the internet (LEONARDO)
export WANDB_MODE=offline

export TOKENIZERS_PARALLELISM=false

# train
python3 -m torch.distributed.run \\
  --nproc_per_node $SLURM_GPUS_ON_NODE \\
  --nnodes $SLURM_NNODES \\
  --node_rank \$SLURM_PROCID \\
  --master_addr $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \\
  --master_port 9901 \\
  /script/gams_sft.py  \\
    --experiment_dir=/output \\
    --model_input_path=$MODEL_INPUT_PATH \\
    --tokenizer_path=$TOKENIZER_PATH \\
    --run_name=${WANDB_RUN_NAME} \\
    --lora_rank=${lora_rank} \\
    --warmup_steps=${warmup_steps} \\
    --learning_rate=${learning_rate} \\
    --min_lr=${min_lr} \\
    $RESUME_ARG

echo \"# completed at \$(date)\"
  """ >> $SCRIPT


# run with singularity
srun \
  --cpu-bind=verbose \
  --output="${EXPERIMENTS_DIR}/${RUN_NAME}.txt" \
  singularity exec \
    --nv \
    -B ${EXPERIMENTS_DIR}:/experiments,${DATA_DIR}:/data,${SCRIPT_DIR}:/script,${OUTPUT_DIR}:/output \
    ${CONTAINER_DIR}/${CONTAINER_FILENAME} \
    /experiments/${RUN_NAME}.sh
