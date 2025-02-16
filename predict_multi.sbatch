#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
#SBATCH --output=logs_predict/slurm_%j.log
#SBATCH --error=logs_predict/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL

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
# You need to tell where predictions and submissions should be saved

# Define an array of data sizes
data_sizes=(2 4 6 8 10)

# Function to process data sizes
process_data_sizes() {
    local data_sizes=("$@")
    local cuda_device=$1
    shift
    for data_size in "${data_sizes[@]}"; do
        echo "Processing data size $data_size"
        ttt_folder=ttt_adapters_llama3/ttt_adapters_llama3_nmax_1500_batch_2_ep_1_lr_5e-5_rank_128_alpha_16.0_${data_size}
        tti_folder=ttt_output_llama3_stages/ttt_adapters_llama3_nmax_1500_batch_2_ep_1_lr_5e-5_rank_128_alpha_16.0_${data_size}
        mkdir -p $tti_folder

        timestamp=$(date '+%Y-%m-%d_%H-%M-%S')

        # measure the time
        start_time=$(date +%s)

        # With lora adapters
        CUDA_VISIBLE_DEVICES=$cuda_device python predict.py \
        --experiment_folder=$tti_folder \
        --pretrained_checkpoint=$base_checkpoint_dir \
        --lora_checkpoints_folder=$ttt_folder \
        --temperature=$temperature \
        --n_sample=$n_sample \
        --data_file=$data_file \
        --solution_file=$solution_file \
        --max_lora_rank=$max_lora_rank \
        --include_n=1 \
        --adapter_number=$data_size \
        --max_tokens=15000 \
        --new_format 

        end_time=$(date +%s)
        echo "Time taken for data_size $data_size: $((end_time - start_time)) seconds"
    done
}

# Run each loop in a separate process
process_data_sizes 0 2 4 6 8 10 &
process_data_sizes 1 20 40 60 80 100 &
process_data_sizes 2 200 400 600 800 1000 &
process_data_sizes 3 1200 1400 1600 1800 2000 &

# Wait for all background processes to complete
wait

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"



