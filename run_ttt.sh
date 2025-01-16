# Specify data path
data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# Specify finetuned path

base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/
# Specify where TTT adapters should be saved
ttt_folder=ttt_adapters
mkdir -p $ttt_folder


# You need show an initial config file that is compatible with torchtune configs
# This is provided in this repo

# lora_config_file=configs/ttt/8.1B_lora_single_device.yaml # for barc
# lora_config_file=configs/ttt/8B_lora_single_device.yaml
lora_config_file=configs/ttt/8B_lora_multi.yaml

# But you can override some of the variables
batch_size=1
epochs=2
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.
num_tasks=2




export MASTER_ADDR=localhost
# export MASTER_ADDR=# Dump memory snapshot history to a file and stop recording
export MASTER_PORT=12345            # Pick a free port
export WORLD_SIZE=4                 # Total number of GPUs
# export RANK=${SLURM_PROCID}
# export LOCAL_RANK=${SLURM_LOCALID}

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"






# You can run the main script
# python test_time_train.py --lora_config=$lora_config_file \
python test_time_train_multi.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--epochs=$epochs \
--num_tasks=${num_tasks} \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--new_format # use --barc_format for barc
