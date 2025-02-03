
source sc_venv_arc/activate.sh
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

# ttt_folder=ttt_adapters/ttt_adapters_0
# ttt_folder=ttt_adapters/ttt_adapters_5
# ttt_folder=ttt_adapters/ttt_adapters_10
# ttt_folder=ttt_adapters/ttt_adapters_50
# ttt_folder=ttt_adapters/ttt_adapters_100
# ttt_folder=ttt_adapters/ttt_adapters_250
# ttt_folder=ttt_adapters/ttt_adapters_500
# ttt_folder=ttt_adapters/ttt_adapters_1000
# ttt_folder=ttt_adapters/ttt_adapters_2000_epochs_1
# ttt_folder=ttt_adapters/ttt_adapters_hard_tasks_2000_epochs_2

# tti_folder=ttt_output_complete/ttt_output_0
# tti_folder=ttt_output_complete/ttt_output_5
# tti_folder=ttt_output_complete/ttt_output_10
# tti_folder=ttt_output_complete/ttt_output_50

# tti_folder=ttt_output_complete/ttt_output_100
# tti_folder=ttt_output_complete/ttt_output_250
# tti_folder=ttt_output_complete/ttt_output_500
# tti_folder=ttt_output_complete/ttt_output_1000
# tti_folder=ttt_output_complete/ttt_output_2000_epochs_1
# tti_folder=ttt_output_complete/ttt_output_hard_tasks_2000_epochs_2

# ttt_folder=ttt_adapters_llama3/ttt_adapters_llama3_10_epochs_1
# tti_folder=ttt_output_llama3/ttt_output_llama3_10_epochs_1

# ttt_folder=ttt_adapters_llama3/ttt_adapters_llama3_40_epochs_1
# tti_folder=ttt_output_llama3/ttt_output_llama3_40_epochs_1

# ttt_folder=ttt_adapters/ttt_adapters_50_epochs_1
# tti_folder=ttt_output_complete/ttt_output_50_epochs_1

# ttt_folder=ttt_adapters/ttt_adapters_40
# tti_folder=ttt_output_complete/ttt_output_40

# ttt_folder=ttt_adapters_llama3/ttt_adapters_llama3_70_epochs_1
# tti_folder=ttt_output_llama3/ttt_output_llama3_70_epochs_1

# ttt_folder=ttt_adapters_llama3/ttt_adapters_llama3_40_epochs_3
# tti_folder=ttt_output_llama3/ttt_output_llama3_40_epochs_3

# Define an array of data sizes
data_sizes=(20 40 60 80)
# data_sizes=(100)

# Loop over each data size
for data_size in "${data_sizes[@]}"; do
    echo "Processing data size $data_size"
    ttt_folder=ttt_adapters_llama3/ttt_adapters_llama3_nmax_1500_batch_2_ep_1_lr_5e-5_rank_128_alpha_16.0_${data_size}
    tti_folder=ttt_output_llama3_stages/ttt_adapters_llama3_nmax_1500_batch_2_ep_1_lr_5e-5_rank_128_alpha_16.0_${data_size}
    mkdir -p $tti_folder

    timestamp=$(date '+%Y-%m-%d_%H-%M-%S')

    # measure the time
    start_time=$(date +%s)

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
    --adapter_number=$data_size \
    --new_format 

    end_time=$(date +%s)
    echo "Time taken for data_size $data_size: $((end_time - start_time)) seconds"
done

