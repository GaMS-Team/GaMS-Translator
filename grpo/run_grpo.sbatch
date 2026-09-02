#!/bin/bash
#SBATCH --job-name=translator_grpo
#SBATCH --partition=boost_usr_prod
#SBATCH --time=4-00
#SBATCH --nodes=3
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=/dev/null
#SBATCH --qos=boost_qos_lprod

# Three nodes are requested: the first two act as trainers, the third serves
# generations over HTTP with `trl vllm-serve`.

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
version=beta-${beta}_r-${lora_rank}_ws-${warmup_steps}_lr-${learning_rate}_min-lr-${min_lr}

# TODO: Add path to the root dir of the GaMS-Translator repository
WORK_DIR=
SCRIPT_DIR=$WORK_DIR/grpo

MODELS_DIR=$WORK_DIR/models

# TODO: Set the name of the DPO-trained model (a dir inside $MODELS_DIR) that GRPO starts from.
# It is produced by merge_lora/run_merge.sbatch after the DPO stage.
INPUT_MODEL_NAME=GaMS3-12B-DPO-Translator
MODEL_INPUT_PATH=$MODELS_DIR/$INPUT_MODEL_NAME
# The merged DPO model carries its own tokenizer, so the tokenizer is read from it too
TOKENIZER_PATH=/model_input_path

# reward models used by reward_functions.py:
#   /reward_models/wmt22-cometkiwi-da/checkpoints/model.ckpt
#   /reward_models/lid.176.bin
REWARD_MODELS_DIR=$WORK_DIR/translation_reward_models

# data: the parquet shards of cjvt/GaMS-Translator-GRPO-Training, fetched by
# download_models.sbatch. The dataset repo keeps them under data/, so the inner
# dir is bound as /data and gams_grpo.py reads /data/{training,validation}-*.parquet
DATA_DIR=$WORK_DIR/data/grpo_training/data

# container
CONTAINER_DIR=$WORK_DIR/singularity
CONTAINER_FILENAME=gams_translator.sif

# set experiment output dir
EXPERIMENTS_DIR=$SCRIPT_DIR/experiments
mkdir -p ${EXPERIMENTS_DIR}/${version}
WANDB_RUN_NAME=GaMS3_12B_GRPO_${version}

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
# /script: '$SCRIPT_DIR'
# /experiments: '$EXPERIMENTS_DIR'
# /data: '$DATA_DIR'
# /output: '$OUTPUT_DIR'
# /reward_models: '$REWARD_MODELS_DIR'
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
export WANDB_PROJECT=GaMS-Translator-GRPO
export WANDB_DIR=/experiments
# Enable offline mode if the GPU partition does not have the access to the internet (LEONARDO)
export WANDB_MODE=offline

export TOKENIZERS_PARALLELISM=false

# 1. Get the list of nodes assigned to this job
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

MASTER_ADDR=\${NODELIST[0]}      # Trainer Node 0
TRAINER_NODE_1=\${NODELIST[0]}
TRAINER_NODE_2=\${NODELIST[1]}
INFERENCE_NODE=\${NODELIST[2]}

echo \"Master Node: \$MASTER_ADDR\"
echo \"Trainer Nodes: \$TRAINER_NODE_1, \$TRAINER_NODE_2\"
echo \"Inference Node: \${INFERENCE_NODE}\"

MY_RANK=\$SLURM_PROCID
MY_HOSTNAME=\$(hostname)
VLLM_PORT=8000

# ------------------------------------------------------------
# ROLE: INFERENCE SERVER (Rank 2)
# ------------------------------------------------------------
if [ \"\$MY_RANK\" -eq 2 ]; then
    echo \"Node \$MY_HOSTNAME, Rank \$MY_RANK starting vLLM Server...\"

    # Start vLLM in the foreground so the SLURM task remains active
    trl vllm-serve \\
        --model /model_input_path \\
        --tensor-parallel-size $SLURM_GPUS_ON_NODE \\
        --port \$VLLM_PORT


# ------------------------------------------------------------
# ROLE: TRAINERS (Ranks 0 and 1)
# ------------------------------------------------------------
elif [ \"\$MY_RANK\" -eq 0 ] || [ \"\$MY_RANK\" -eq 1 ]; then
    echo \"Node \$MY_HOSTNAME, Rank \$MY_RANK acting as Trainer.\"

    VLLM_URL=\"http://\${INFERENCE_NODE}:\$VLLM_PORT\"
    echo \"Inference URL: \$VLLM_URL\"

    # train
    python3 -m torch.distributed.run \\
      --nproc_per_node $SLURM_GPUS_ON_NODE \\
      --nnodes 2 \\
      --node_rank \$MY_RANK \\
      --master_addr \$MASTER_ADDR \\
      --master_port 9901 \\
      /script/gams_grpo.py  \\
        --experiment_dir=/output \\
        --model_input_path=/model_input_path \\
        --tokenizer_path=$TOKENIZER_PATH \\
        --run_name=${WANDB_RUN_NAME} \\
        --vllm_url=\${VLLM_URL} \\
        --lora_rank=${lora_rank} \\
        --warmup_steps=${warmup_steps} \\
        --learning_rate=${learning_rate} \\
        --min_lr=${min_lr} \\
        --beta=${beta} \\
        $RESUME_ARG

    echo \"# completed at \$(date)\"
    # training is done, so release the vLLM server node as well
    scancel $SLURM_JOB_ID
fi
  """ >> $SCRIPT


# run with singularity
srun \
  --cpu-bind=verbose \
  --output="${EXPERIMENTS_DIR}/${RUN_NAME}.txt" \
  singularity exec \
    --nv \
    -B ${EXPERIMENTS_DIR}:/experiments,${DATA_DIR}:/data,${MODEL_INPUT_PATH}:/model_input_path,${SCRIPT_DIR}:/script,${OUTPUT_DIR}:/output,${REWARD_MODELS_DIR}:/reward_models \
    ${CONTAINER_DIR}/${CONTAINER_FILENAME} \
    /experiments/${RUN_NAME}.sh
