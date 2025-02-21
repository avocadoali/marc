#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --account=hai_hreplearn
#SBATCH --cpus-per-task=12
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
# You need to tell where predictions and submissions should be saved

logs_predict_folder=logs_predict/multi_batch
mkdir -p $logs_predict_folder

# Function to process data sizes
process_data_sizes() {
    local data_sizes=("$@")
    local cuda_device=$1
    shift

    echo "cuda_device: $cuda_device"
    echo "data_sizes: ${data_sizes[@]:1}"
    
    for data_size in "${data_sizes[@]:1}"; do
        echo "Processing data size $data_size"
        ttt_folder=experiments_thesis/2k_test_run/adapters_json_${data_size}
        tti_folder=experiments_thesis/output_2k_test_run/adapters_json_${data_size}
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
        --new_format | tee $logs_predict_folder/slurm_${data_size}.log

        end_time=$(date +%s)
        echo "Time taken for data_size $data_size: $((end_time - start_time)) seconds"
        # echo current  time date
        echo "End time: $(date +%Y-%m-%d_%H-%M-%S)"

    done
}

# Run each loop in a separate process
# process_data_sizes 0  10 20 

# process_data_sizes 0  2 20 200 1200 &
# process_data_sizes 1  4 40 400 10   &
# process_data_sizes 2  6 60 600 100  &
# process_data_sizes 3  8 80 800 1000 &



process_data_sizes 0  2  10 &
process_data_sizes 1  4  20 &
process_data_sizes 2  6  34 &
process_data_sizes 3  8  &




# Wait for all background processes to complete
wait

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"



