#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs_ttt/slurm_%j.log
#SBATCH --error=logs_ttt/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL

# load the environment

source sc_venv_arc/activate.sh

# Set up distributed training environment variables
# export RANK=${SLURM_PROCID}
# export LOCAL_RANK=${SLURM_LOCALID}
# export WORLD_SIZE=${SLURM_NTASKS}
# # export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# # export MASTER_ADDR=$(srun hostname --ip-address | head -n 1)
# export MASTER_ADDR=# Dump memory snapshot history to a file and stop recording
# export MASTER_PORT=12345            # Pick a free port
# export WORLD_SIZE=4                 # Total number of GPUs
# echo "Master address: $MASTER_ADDR"
# echo "Master port: $MASTER_PORT"
# echo "World size: $WORLD_SIZE"
# echo "Local rank: $LOCAL_RANK"
# echo "Rank: $RANK"



# Pytorch memory allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


./run_ttt.sh
echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"
