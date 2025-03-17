#!/usr/bin/env bash

#SBATCH --job-name=predict-baseline-ekin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=4:00:00
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
# base_checkpoint_dir=/fs/work/workspace/scratch/tum_ind3695-arc-workspace/huggingface/hub/models--ekinakyurek--Llama-3.1-ARC-Potpourri-Transduction-8B/snapshots/1421b20207ad9af480aff4dccde964af8d64203e
base_checkpoint_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
# ttt_folder=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2

# if solution file is given predict will evaluate the model
solution_file=arc-prize-2024/arc-agi_evaluation_solutions.json

temperature=0
n_sample=1

# this should be same as your ttt
max_lora_rank=128

# Set up folders
# ttt_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/huggingface/hub/models--ekinakyurek--marc-lora-adapters-Llama-3.1-ARC-Potpourri-Transduction-8B/snapshots/528591c27f3332497e3f4cbbf9eae39f3422a066
# tti_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_barc_baseline
# mkdir -p $tti_folder

# /hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/baseline_barc/baseline_barc/adapters_json
# ttt_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/baseline_barc/baseline_barc/adapters_json
# tti_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/baseline_barc/baseline_barc_output
# mkdir -p $tti_folder

# /hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/baseline_ekin/baseline_ekin/adapters_json
ttt_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/baseline_ekin/baseline_ekin/adapters_json
tti_folder=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/baseline_ekin/baseline_ekin_output
mkdir -p $tti_folder


# With lora adapters
python predict.py \
--experiment_folder=$tti_folder \
--pretrained_checkpoint=$base_checkpoint_dir \
--lora_checkpoints_folder=$ttt_folder \
--temperature=$temperature \
--n_sample=$n_sample \
--data_file=$data_file \
--solution_file=$solution_file \
--max_lora_rank=$max_lora_rank \
--include_n=1 \
--num_gpus=4 \
--barc_format | tee $tti_folder/logs.log &


# Wait for all background processes to complete
wait

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"

