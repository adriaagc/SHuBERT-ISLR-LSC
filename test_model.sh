#!/bin/bash
#SBATCH --job-name=eval_1B_full_finetune
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/agarciac/logs/1B_eval/2026-08-15_17-59/slurm_%x_%j.out
#SBATCH --error=/home/agarciac/logs/1B_eval/2026-08-15_17-59/slurm_%x_%j.err

export PYTHONWARNINGS="ignore"
export HYDRA_FULL_ERROR=1

# SHuBERT repo root
cd /home/agarciac/code/SHuBERT
PROJECT_ROOT=$PWD

module load conda
conda activate work_env

fairseq_root=$PROJECT_ROOT/fairseq
code_dir=$fairseq_root/examples/shubert

export PYTHONPATH="$fairseq_root:$fairseq_root/examples:$PYTHONPATH"

# not uploading to wandb (can do it later)

# --- CHANGE THE PATH DEPENDING ON THE MODEL WE EVALUATE ---

# Route to the checkpoint we want to eval (best or another one)
# TO CHANGE: <TIMESTAMP> and name of the .pt:
CKPT=/home/agarciac/logs/1B/ckpt/2026-08-15_17-59/checkpoint_best.pt

# Split to eval: test.tsv and test.gloss already in signer_disjoint
SUBSET=test
DATA_DIR=/data/upftfg41/shared/data/LSC_Corpus_processed/shubert_v1_features/islr_manifests/signer_disjoint
BATCH_SIZE=128

# -----------------------------------------------------------

# change also according to the task to evaluate (e.g. 1A, 1B, etc)
log_dir=/home/agarciac/logs/1B_eval
mkdir -p $log_dir

# create a folder that has the same "name/date" as the checkpoint we are evaluating.
folder_dir=$log_dir/2026-08-15_17-59
mkdir -p $folder_dir

# we remove --fp16 \ as in validation there is no backward pass and no gradients are stored.

srun --nodes=1 --ntasks=1 fairseq-validate $DATA_DIR \
        --user-dir $code_dir \
        --task shubert_islr \
        --path $CKPT \
        --valid-subset $SUBSET \
        --batch-size $BATCH_SIZE \
        --log-format json \
        > $folder_dir/eval_log.txt 2>&1