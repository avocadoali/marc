#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs_predict/slurm_%j.log
#SBATCH --error=logs_predict/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL


source sc_venv_inference/activate.sh

echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"
# train
./run_predict.sh
echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"
