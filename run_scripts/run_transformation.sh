# load the environment

# source sc_venv_arc/activate.sh

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
start_time=$(date +%s)
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
experiment_name="baseline_250_permute_1_ep_1"
ttt_folder="experiments_thesis/${experiment_name}/adapters_json"
mkdir -p $ttt_folder

python debug_transformations.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--offset=0 \
--num_tasks=400 \
--Nmax=$nmax \
--permute_n=1 \
--epochs=$epochs \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--experiment_name=$experiment_name \
--new_format 

echo "Done at $(date +%Y-%m-%d_%H-%M-%S)"
# time taken
time_taken=$(($(date +%s) - start_time))
echo "Time taken: $time_taken seconds"
