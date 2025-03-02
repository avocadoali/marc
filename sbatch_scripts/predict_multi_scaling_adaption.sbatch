#!/usr/bin/env bash

#SBATCH --job-name=predict_scaling_adaption
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --partition=dev_accelerated
#SBATCH --account=hk-project-pai00039
#SBATCH --output=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --error=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/logs_default/slurm_%j.log
#SBATCH --mail-user=avocadoaling@gmail.com
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate vllm_marc

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
echo "VLLM_ALLOW_LONG_MAX_MODEL_LEN: $VLLM_ALLOW_LONG_MAX_MODEL_LEN"

module load compiler/gnu/11

echo "starting ttt at time $(date +%Y-%m-%d_%H-%M-%S)"
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
# You need to tell where predictions and submissions should be saved


# scratch dir
# scratch_dir=/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace
# # location of generated adapaters
# ttt_experiment_folder="${scratch_dir}/experiments_thesis_dataset_scaling"
# experiment_name="output_1000_permute_3-4_20k_double"
# ttt_folder="${ttt_experiment_folder}/${experiment_name}/adapters_json"

ttt_adapters_folder="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_dataset_scaling/1000_permute_3-4_20k_double"
output_folder="${ttt_adapters_folder}_output"
mkdir -p $output_folder

echo "ttt_adapters_folder: ${ttt_adapters_folder}"
echo "output_folder: ${output_folder}"

# Function to process data sizes
process_data_sizes() {
    local data_sizes=("$@")
    local cuda_device=$1
    shift

    echo "cuda_device: $cuda_device"
    echo "data_sizes: ${data_sizes[@]:1}"
    
    for data_size in "${data_sizes[@]:1}"; do
        echo "Processing data size $data_size"
        ttt_folder=${ttt_adapters_folder}/adapters_json_${data_size}
        tti_folder=${output_folder}/adapters_json_${data_size}
        mkdir -p $tti_folder

        echo "ttt_folder: $ttt_folder"
        echo "tti_folder: $tti_folder"

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
        --max_tokens=20000 \
        --new_format | tee $output_folder/slurm_device_${cuda_device}.log

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



process_data_sizes 0  2 20 200 10   &
process_data_sizes 1  4 40 400 100  &
process_data_sizes 2  6 60 600 1000 &
process_data_sizes 3  8 80 800 &




# Wait for all background processes to complete
wait

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"



