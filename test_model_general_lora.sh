#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00

# --- MANUAL CONFIGURATION ---
JOB_NAME="eval_1B_full_finetune_spontaneous" # "eval_1B_full_finetune" or "eval_1A_frozen_encoder"
DATETIME="2026-10-29_21-41"
TASK_NAME="1B" #"1B"
CKPT_FILE="checkpoint_best.pt"
BATCH_SIZE=128
SUBSET="test"

PROJECT_ROOT="/home/agarciac/code/SHuBERT"
BASE_LOG_DIR="/home/agarciac/logs"
DATA_DIR="/data/upftfg41/shared/data/Spontaneous-LSC-Recognition/shubert_v1_features/islr_manifests/signer_disjoint"
# ----------------------------

# job-name:
scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME

# Routes:
EVAL_LOG_DIR="${BASE_LOG_DIR}/${TASK_NAME}_eval/${DATETIME}"
CKPT_PATH="${BASE_LOG_DIR}/${TASK_NAME}/ckpt/${DATETIME}/${CKPT_FILE}"

mkdir -p "$EVAL_LOG_DIR"

# .out and .err routes:
exec > "${EVAL_LOG_DIR}/slurm_${JOB_NAME}_${SLURM_JOB_ID}.out" 2> "${EVAL_LOG_DIR}/slurm_${JOB_NAME}_${SLURM_JOB_ID}.err"

export PYTHONWARNINGS="ignore"
export HYDRA_FULL_ERROR=1

cd "$PROJECT_ROOT"

module load conda
conda activate work_env

fairseq_root="${PROJECT_ROOT}/fairseq"
code_dir="${fairseq_root}/examples/shubert"

export PYTHONPATH="${fairseq_root}:${fairseq_root}/examples:${PYTHONPATH}"

srun --nodes=1 --ntasks=1 fairseq-validate "$DATA_DIR" \
        --user-dir "$code_dir" \
        --task shubert_islr \
        --path "$CKPT_PATH" \
        --valid-subset "$SUBSET" \
        --batch-size $BATCH_SIZE \
        --log-format json \
        --model-overrides " {'model': {'_name': 'shubert_lora'}} " \
        > "${EVAL_LOG_DIR}/eval_log.txt" 2>&1