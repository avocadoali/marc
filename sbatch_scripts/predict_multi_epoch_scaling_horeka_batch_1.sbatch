#!/usr/bin/env bash

#SBATCH --job-name=predict_scaling_epoch_batch_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --time=13:00:00
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-pai00039
#SBATCH --output=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --error=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL


source ~/.bashrc
conda activate vllm_marc

module load compiler/gnu/11

echo "starting predict at time $(date +%Y-%m-%d_%H-%M-%S)"
# train

# Specify data path
data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# Tell where your Fintuned (named as base) and TTT checkpoints are

# base_checkpoint_dir=/path/to/finetuned/model/folder/
# ttt_folder=/path/to/ttt/folder
# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
base_checkpoint_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
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


adapters_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_1
output_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_1_output
output_log_folder=${output_folder}/logs
mkdir -p $output_log_folder                                                                                     

###
# Batch 1, Epoch 1
###
# experiments dirs (input/ouput)
ttt_folder=${adapters_folder}/adapters_json_ep_0_iter_-1
tti_folder=${output_folder}/adapters_json_ep_0_iter_-1
mkdir -p $tti_folder

echo "Batch 1, Epoch 1" | tee $output_log_folder/slurm_epoch_0.log
echo "ttt_folder: ${ttt_folder}"
echo "tti_folder: ${tti_folder}"

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
--new_format | tee $output_log_folder/slurm_epoch_0.log &


###
# Batch 1, Epoch 2
###
ttt_folder=${adapters_folder}/adapters_json_ep_1_iter_-1
tti_folder=${output_folder}/adapters_json_ep_1_iter_-1
mkdir -p $tti_folder

echo "Batch 1, Epoch 2" | tee $output_log_folder/slurm_epoch_1.log

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
--new_format | tee $output_log_folder/slurm_epoch_1.log &


###
# Batch 1, Epoch 3
###
ttt_folder=${adapters_folder}/adapters_json_ep_2_iter_-1
tti_folder=${output_folder}/adapters_json_ep_2_iter_-1
mkdir -p $tti_folder

echo "Batch 1, Epoch 3" | tee $output_log_folder/slurm_epoch_2.log

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
--new_format | tee $output_log_folder/slurm_epoch_2.log &


# Wait for all background processes to complete
wait

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"

