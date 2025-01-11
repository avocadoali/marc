#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.log

# load the environment
pwd


source sc_venv_inference/activate.sh

# train
./run_predict.sh



