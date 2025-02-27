#!/usr/bin/env bash


#SBATCH --job-name=ttt-scaling-adaption
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --partition=dev_accelerated-h100
#SBATCH --account=hk-project-pai00039
#SBATCH --output=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --error=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL


# load the environment

source sc_venv_arc/activate.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"

data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# data_file=arc-prize-2024/arc-hard-tasks.json
# data_file=arc-prize-2024/arc-unsolved-easy-tasks.json
# Specify finetuned path

# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/24ae87a9c340aa4207dd46509414c019998e0161
# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
base_checkpoint_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a

## Llama3 8b instruct base 
# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/blobs/4c763399690c7fec642231350756e4ee9184b5ce

# base_checkpoint_dir= /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--barc0--Llama-3.1-ARC-Potpourri-Transduction-8B/snapshots/f4c4f24c9428d051748af8049099132171115009
# Specify where TTT adapters should be saved

# You need show an initial config file that is compatible with torchtune configs
# This is provided in this repo

# lora_config_file=configs/ttt/8.1B_lora_single_device.yaml # for barc
lora_config_file=configs/ttt/8B_lora_single_device.yaml
# lora_config_file=configs/ttt/DeepSeek_r1_8B_lora_single_device.yaml
# lora_config_file=configs/ttt/8B_lora_multi.yaml

# But you can override some of the variables
batch_size=1
epochs=1
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.

nmax=1000
# Automatically construct folder name from variables
# ttt_folder="experiments/ttt_adapters_nmax_${nmax}_batch_${batch_size}_ep_${epochs}_lr_${learning_rate}_rank_${lora_rank}_alpha_${lora_alpha}"


# scratch dir
scratch_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace
# location of generated adapaters
ttt_experiment_folder="${scratch_dir}/experiments_thesis_dataset_scaling"
experiment_name="1000_permute_3-4_20k_double"
ttt_folder="${ttt_experiment_folder}/${experiment_name}/adapters_json"
ttt_log_folder="${ttt_experiment_folder}/${experiment_name}/logs"

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
--new_format | tee $ttt_log_folder/log_0.log &

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
--new_format | tee $ttt_log_folder/log_1.log &

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
--new_format | tee $ttt_log_folder/log_2.log &

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
--new_format | tee $ttt_log_folder/log_3.log &

# Wait for all background processes to complete
wait

echo "All tasks completed at $(date +%Y-%m-%d_%H-%M-%S)"
