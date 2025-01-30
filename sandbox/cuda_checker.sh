CUDA_VISIBLE_DEVICES='-1'

python sandbox.py



# But you can override some of the variables
batch_size=2
epochs=1
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.

nmax=100

# Automatically construct folder name from variables
ttt_folder="ttt_adapters_llama3/ttt_adapters_llama3_nmax_${nmax}_batch_${batch_size}_ep_${epochs}_lr_${learning_rate}_rank_${lora_rank}_alpha_${lora_alpha}"


echo $ttt_folder