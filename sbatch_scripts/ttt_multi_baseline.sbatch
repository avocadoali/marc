#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs_ttt/slurm_%j.log
#SBATCH --error=logs_ttt/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --job-name=ttt-baseline-100-permute-1-ep-1

# load the environment


source sc_venv_arc/activate.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"

data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# data_file=arc-prize-2024/arc-hard-tasks.json
# data_file=arc-prize-2024/arc-unsolved-easy-tasks.json
# Specify finetuned path

# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/24ae87a9c340aa4207dd46509414c019998e0161
base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
# base_checkpoint_dir= /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--barc0--Llama-3.1-ARC-Potpourri-Transduction-8B/snapshots/f4c4f24c9428d051748af8049099132171115009
# Specify where TTT adapters should be saved

# You need show an initial config file that is compatible with torchtune configs
# This is provided in this repo

# lora_config_file=configs/ttt/8.1B_lora_single_device.yaml # for barc
lora_config_file=configs/ttt/8B_lora_single_device.yaml
# lora_config_file=configs/ttt/DeepSeek_r1_8B_lora_single_device.yaml
# lora_config_file=configs/ttt/8B_lora_multi.yaml

# But you can override some of the variables
batch_size=2
epochs=1
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.

nmax=250
# Automatically construct folder name from variables
# ttt_folder="experiments/ttt_adapters_nmax_${nmax}_batch_${batch_size}_ep_${epochs}_lr_${learning_rate}_rank_${lora_rank}_alpha_${lora_alpha}"

## dont forget to copy the adapters_json folder to the new experiment folder
ttt_experiment_name="baseline_250_permute_1_ep_1"

ttt_log_folder="experiments_thesis/${ttt_experiment_name}/logs"
ttt_folder="experiments_thesis/${ttt_experiment_name}/adapters_json"

echo "running experiment ${ttt_experiment_name}"
echo "log folder: ${ttt_log_folder}"
echo "adapter folder: ${ttt_folder}"

mkdir -p $ttt_folder
mkdir -p $ttt_log_folder

CUDA_VISIBLE_DEVICES=0 python test_time_train.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--offset=0 \
--num_tasks=100 \
--Nmax=$nmax \
--epochs=$epochs \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--new_format | tee $ttt_log_folder/log_0.txt &

CUDA_VISIBLE_DEVICES=1 python test_time_train.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--offset=100 \
--num_tasks=100 \
--Nmax=$nmax \
--epochs=$epochs \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--new_format | tee $ttt_log_folder/log_1.txt &

CUDA_VISIBLE_DEVICES=2 python test_time_train.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--offset=200 \
--num_tasks=100 \
--Nmax=$nmax \
--epochs=$epochs \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--new_format | tee $ttt_log_folder/log_2.txt &

CUDA_VISIBLE_DEVICES=3 python test_time_train.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--offset=300 \
--num_tasks=100 \
--Nmax=$nmax \
--epochs=$epochs \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--new_format | tee $ttt_log_folder/log_3.txt &

# Wait for all background processes to complete
wait

echo "experiment ${ttt_experiment_name} completed"
echo "All tasks completed at $(date +%Y-%m-%d_%H-%M-%S)"