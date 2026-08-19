#!/bin/bash
#SBATCH --job-name=
#SBATCH --dependency=
#SBATCH --partition=
#SBATCH --ntasks-per-node=8
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out
#SBATCH --error=/path_to_output_logs_dir/slurm_%x.err


# the gpu nodes of Pirineus III only have 2 GPUs H100 per node. Therefore, we cannot ask for 8 GPUs in a single node. 

ulimit -n 65535 

# number of individual processes:
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
export NCCL_TIMEOUT=1200


cd fairseq

CONDA_ROOT=
env_name=
fairseq_root=fairseq
path=fairseq/examples/shubert/config    # route where the .yaml is 
config=base_random                      # name of the .yaml file
code_dir=fairseq/examples/shubert       # dir where shubert.py and shubert_isrl.py are  


# loading conda environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $env_name

# create log dir
log_dir=/path_to_output_logs_dir/log_dir_name
mkdir -p $log_dir


PYTHONPATH=$PYTHONPATH:$fairseq_root/examples \
        srun fairseq-hydra-train  \
        --config-dir $path \
        --config-name $config \
        common.user_dir=$code_dir \
        checkpoint.save_dir=$log_dir/ckpt \
        common.log_file=$log_dir/log.txt \
        common.tensorboard_logdir=$log_dir/tb \
        distributed_training.distributed_world_size=$WORLD_SIZE

