# Specify data path
data_file=arc-prize-2024/arc-agi_evaluation_challenges.json
# Specify finetuned path

base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a
# base_checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/
# Specify where TTT adapters should be saved

# You need show an initial config file that is compatible with torchtune configs
# This is provided in this repo

# lora_config_file=configs/ttt/8.1B_lora_single_device.yaml # for barc
lora_config_file=configs/ttt/8B_lora_single_device.yaml
# lora_config_file=configs/ttt/8B_lora_multi.yaml

# But you can override some of the variables
batch_size=2
epochs=2
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.

# export MASTER_ADDR=localhost
# # export MASTER_ADDR=# Dump memory snapshot history to a file and stop recording
# export MASTER_PORT=29601    # Pick a free port
# export WORLD_SIZE=4                 # Total number of GPUs
# # set some unique rank for each process
# export RANK=0
# # export LOCAL_RANK=${SLURM_LOCALID}

# echo "Master address: $MASTER_ADDR"
# echo "Master port: $MASTER_PORT"
# echo "World size: $WORLD_SIZE"
# echo "Rank: $RANK"

# ttt_folder=ttt_adapters_5
# mkdir -p $ttt_folder
# python test_time_train.py --lora_config=$lora_config_file \
# --base_checkpoint_dir=$base_checkpoint_dir \
# --experiment_folder=$ttt_folder \
# --data_file=$data_file \
# --batch_size=$batch_size \
# --Nmax=5 \
# --epochs=$epochs \
# --lora_rank=$lora_rank \
# --lora_alpha=$lora_alpha \
# --lora_to_output=$lora_to_output \
# --new_format 

# ttt_folder=ttt_adapters_10
# mkdir -p $ttt_folder
# python test_time_train.py --lora_config=$lora_config_file \
# --base_checkpoint_dir=$base_checkpoint_dir \
# --experiment_folder=$ttt_folder \
# --data_file=$data_file \
# --batch_size=$batch_size \
# --Nmax=10 \
# --epochs=$epochs \
# --lora_rank=$lora_rank \
# --lora_alpha=$lora_alpha \
# --lora_to_output=$lora_to_output \
# --new_format 

# ttt_folder=ttt_adapters_50
# # mkdir -p $ttt_folder
# python test_time_train.py --lora_config=$lora_config_file \
# --base_checkpoint_dir=$base_checkpoint_dir \
# --experiment_folder=$ttt_folder \
# --data_file=$data_file \
# --batch_size=$batch_size \
# --Nmax=50 \
# --epochs=$epochs \
# --lora_rank=$lora_rank \
# --lora_alpha=$lora_alpha \
# --lora_to_output=$lora_to_output \
# --new_format 

# ttt_folder=ttt_adapters_100
# mkdir -p $ttt_folder
# python test_time_train.py --lora_config=$lora_config_file \
# --base_checkpoint_dir=$base_checkpoint_dir \
# --experiment_folder=$ttt_folder \
# --data_file=$data_file \
# --batch_size=$batch_size \
# --Nmax=100 \
# --epochs=$epochs \
# --lora_rank=$lora_rank \
# --lora_alpha=$lora_alpha \
# --lora_to_output=$lora_to_output \
# --new_format 
plt.show()
# ttt_folder=ttt_adapters_250
# mkdir -p $ttt_folder
# python test_time_train.py --lora_config=$lora_config_file \
# --base_checkpoint_dir=$base_checkpoint_dir \
# --experiment_folder=$ttt_folder \
# --data_file=$data_file \
# --batch_size=$batch_size \
# --Nmax=250 \
# --epochs=$epochs \
# --lora_rank=$lora_rank \
# --lora_alpha=$lora_alpha \
# --lora_to_output=$lora_to_output \
# --new_format 

ttt_folder=ttt_adapters_500
mkdir -p $ttt_folder
python test_time_train.py --lora_config=$lora_config_file \
--base_checkpoint_dir=$base_checkpoint_dir \
--experiment_folder=$ttt_folder \
--data_file=$data_file \
--batch_size=$batch_size \
--Nmax=500 \
--epochs=$epochs \
--lora_rank=$lora_rank \
--lora_alpha=$lora_alpha \
--lora_to_output=$lora_to_output \
--new_format 

# ttt_folder=ttt_adapters_1000
# mkdir -p $ttt_folder
# python test_time_train.py --lora_config=$lora_config_file \
# --base_checkpoint_dir=$base_checkpoint_dir \
# --experiment_folder=$ttt_folder \
# --data_file=$data_file \
# --batch_size=$batch_size \
# --offset=100 \
# --num_tasks=100 \
# --Nmax=1000 \
# --epochs=$epochs \
# --lora_rank=$lora_rank \
# --lora_alpha=$lora_alpha \
# --lora_to_output=$lora_to_output \
# --new_format 


