#!/bin/bash
#SBATCH --job-name=1A_frozen_shubert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=3:00:00
#SBATCH --output=/home/agarciac/logs/1A/slurm_%x_%j.out     #/path_to_output_logs_dir/slurm_%x.out
#SBATCH --error=/home/agarciac/logs/1A/slurm_%x_%j.err      #/path_to_output_logs_dir/slurm_%x.err

# ulimit -n 65535

# $SLURM_JOB_NUM_NODES <-- nodes: in each node of the gpu partition we have 2 GPUs H100
# $SLURM_NTASKS_PER_NODE <-- ntasks-per-node: therefore 2 processes.

export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * 1)) # nodes * tasks_per_node
export NCCL_TIMEOUT=1200
export PYTHONWARNINGS="ignore"
export HYDRA_FULL_ERROR=1

# enter the root dir of the SHuBERT repo: all the routes will be relative to this dir.
cd /home/agarciac/code/SHuBERT   #fairseq
# absolute route
PROJECT_ROOT=$PWD

module load conda
conda activate work_env


fairseq_root=$PROJECT_ROOT/fairseq
path=$fairseq_root/examples/shubert/config              # route where the .yaml is 
config=islr_frozen_shubert_encoder                      # name of the .yaml file
code_dir=$fairseq_root/examples/shubert                 # dir where shubert.py and shubert_isrl.py are  


# create log dir
log_dir=/home/agarciac/logs/1A #/path_to_output_logs_dir/log_dir_name
mkdir -p $log_dir
mkdir -p $log_dir/ckpt

export PYTHONPATH="$fairseq_root:$fairseq_root/examples:$PYTHONPATH"

export WANDB_MODE=offline
export WANDB_API_KEY="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"
export WANDB_ENTITY="agarciaac-universitat-pompeu-fabra"
export WANDB_PROJECT="shubert_islr"
export WANDB_NAME="islr_shubert_frozen_encoder" # this is the run name
export WANDB_RUN_GROUP="1A_head_only"  # to organise the wandb workspace in groups

TIMESTAMP=$(date +%Y-%m-%d_%H-%M)

#common.tensorboard_logdir=$log_dir/tb \
#PYTHONPATH=$PYTHONPATH:$fairseq_root/examples \
srun --nodes=1 --ntasks=1 fairseq-hydra-train  \
        --config-dir $path \
        --config-name $config \
        common.user_dir=$code_dir \
        common.wandb_project="shubert_islr" \
        checkpoint.save_dir=$log_dir/ckpt/$TIMESTAMP \
        common.log_file=$log_dir/log.txt \
        distributed_training.distributed_world_size=$WORLD_SIZE

