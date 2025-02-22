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

cd /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/torchtune


tune run --nproc_per_node 4 lora_finetune_distributed --config configs/ttt/8B_lora_multi.yaml checkpointer.checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/ > logs_ttt/torchtune_distributed_test.log
echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"
