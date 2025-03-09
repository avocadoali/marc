#!/usr/bin/env bash

#SBATCH --job-name=ttt-ekin-recreate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --mem-per-gpu=193000
#SBATCH --partition=dev_accelerated
#SBATCH --account=hk-project-pai00039
#SBATCH --output=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --error=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL

# load the environment
conda activate marc


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"


# Specify data path
data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# Specify finetuned path
base_checkpoint_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a

# You need show an initial config file that is compatible with torchtune configs
# This is provided in this repo
lora_config_file=configs/ttt/8B_lora_single_device.yaml
# lora_config_file=configs/ttt/8.1B_lora_single_device.yaml # for barc
# But you can override some of the variables
batch_size=2
epochs=2
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.
num_tasks=15

nmax=250

# Specify where TTT adapters should be saved
# scratch dir
scratch_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace
# location of generated adapaters
# /hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling_8/epoch_scaling_ekin
ttt_experiment_folder="${scratch_dir}/experiments_ekin_recreate"
experiment_name="ekin_recreate"
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

