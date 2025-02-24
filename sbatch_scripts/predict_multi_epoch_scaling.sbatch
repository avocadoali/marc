#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=8
#SBATCH --output=logs_predict/slurm_%j.log
#SBATCH --error=logs_predict/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --job-name=predict_multi_1-1200
#SBATCH --mail-type=ALL

source sc_venv_inference/activate.sh

echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"
# train

# Specify data path
data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# Tell where your Fintuned (named as base) and TTT checkpoints are

# base_checkpoint_dir=/path/to/finetuned/model/folder/
# ttt_folder=/path/to/ttt/folder
base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
# ttt_folder=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2

# if solution file is given predict will evaluate the model
solution_file=arc-prize-2024/arc-agi_evaluation_solutions.json

temperature=0
n_sample=1

# this should be same as your ttt
max_lora_rank=128

# Slurm logs
logs_predict_folder=logs_predict/multi_batch
mkdir -p $logs_predict_folder



###
# Batch 1, Epoch 1
###
# experiments dirs (input/ouput)
experiment_name=experiments_thesis/baseline_250_permute_1_8k_batch_1_ep_1
ttt_folder=${experiment_name}/adapters_json
tti_folder=${experiment_name}_output/
mkdir -p $tti_folder


echo "Batch 1, Epoch 1" | tee $tti_folder/slurm_0.log
# With lora adapters
CUDA_VISIBLE_DEVICES=0 python predict.py \
--experiment_folder=$tti_folder \
--pretrained_checkpoint=$base_checkpoint_dir \
--lora_checkpoints_folder=$ttt_folder \
--temperature=$temperature \
--n_sample=$n_sample \
--data_file=$data_file \
--solution_file=$solution_file \
--max_lora_rank=$max_lora_rank \
--include_n=1 \
--new_format | tee $tti_folder/slurm_0.log &

###
# Batch 2, Epoch 1
###

experiment_name=experiments_thesis/baseline_250_permute_1_8k_batch_2_ep_1
ttt_folder=${experiment_name}/adapters_json
tti_folder=${experiment_name}_output/
mkdir -p $tti_folder

echo "Batch 2, Epoch 1" | tee $tti_folder/slurm_1.log


CUDA_VISIBLE_DEVICES=1 python predict.py \
--experiment_folder=$tti_folder \
--pretrained_checkpoint=$base_checkpoint_dir \
--lora_checkpoints_folder=$ttt_folder \
--temperature=$temperature \
--n_sample=$n_sample \
--data_file=$data_file \
--solution_file=$solution_file \
--max_lora_rank=$max_lora_rank \
--include_n=1 \
--new_format | tee $tti_folder/slurm_1.log &

###
# Batch 1, Epoch 2
###

experiment_name=experiments_thesis/baseline_250_permute_1_8k_batch_1_ep_2
ttt_folder=${experiment_name}/adapters_json
tti_folder=${experiment_name}_output/
mkdir -p $tti_folder

echo "Batch 1, Epoch 2" | tee $tti_folder/slurm_2.log

CUDA_VISIBLE_DEVICES=2 python predict.py \
--experiment_folder=$tti_folder \
--pretrained_checkpoint=$base_checkpoint_dir \
--lora_checkpoints_folder=$ttt_folder \
--temperature=$temperature \
--n_sample=$n_sample \
--data_file=$data_file \
--solution_file=$solution_file \
--max_lora_rank=$max_lora_rank \
--include_n=1 \
--new_format | tee $tti_folder/slurm_2.log &

###
# Batch 2, Epoch 2
###

experiment_name=experiments_thesis/baseline_250_permute_1_8k_batch_2_ep_2
ttt_folder=${experiment_name}/adapters_json
tti_folder=${experiment_name}_output/
mkdir -p $tti_folder

echo "Batch 2, Epoch 2" | tee $tti_folder/slurm_3.log

CUDA_VISIBLE_DEVICES=3 python predict.py \
--experiment_folder=$tti_folder \
--pretrained_checkpoint=$base_checkpoint_dir \
--lora_checkpoints_folder=$ttt_folder \
--temperature=$temperature \
--n_sample=$n_sample \
--data_file=$data_file \
--solution_file=$solution_file \
--max_lora_rank=$max_lora_rank \
--include_n=1 \
--new_format | tee $tti_folder/slurm_3.log &

# Wait for all background processes to complete
wait

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"

