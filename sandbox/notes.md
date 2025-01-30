
    "output_directory": "ttt_adapters_llama3/ttt_adapters_llama3_70_epochs_1",

"output_directory": "ttt_adapters_llama3/ttt_adapters_llama3_40_epochs_3",




    "output_directory": "ttt_adapters_llama3/ttt_adapters_llama3_10_epochs_1",





    "output_directory": "ttt_adapters_llama3/ttt_adapters_llama3_50_epochs_2_lora_alpha_32",


    "output_directory": "ttt_adapters_llama3/ttt_adapters_llama3_50_epochs_3",


    "output_directory": "ttt_adapters_llama3/ttt_adapters_llama3_50_epochs_3_lora_alpha_32",


# But you can override some of the variables
batch_size=2
epochs=1
learning_rate=5e-5
lora_rank=128
lora_alpha=16.0
lora_to_output=False # doesn't apply for Llama3.2 models for now.
# You can specify how many tasks you want train for.

nmax=100



## TTT
| Job Name | nmax | epochs | batch_size | learning_rate | lora_rank | lora_alpha | lora_to_output | done |
| -------- | ---- | ------ | ---------- | ------------- | --------- | ---------- | -------------- | ---- |
|          | 2    | 1      | 5e-5       | 128           | 16.0      | False      | 100            |      |




## Predict

| Job Name                                            | jobid    | nmax | epochs | batch_size | learning_rate | lora_rank | lora_alpha | lora_to_output | done    |
| --------------------------------------------------- | -------- | ---- | ------ | ---------- | ------------- | --------- | ---------- | -------------- | ------- |
| ttt_adapters_llama3/ttt_adapters_llama3_40_epochs_3 | 10886601 | 40   | 3      | 2          | 5e-5          | 128       | 16.0       | False          | ongoing |
| ttt_adapters_llama3/ttt_adapters_llama3_50_epochs_1 | 10880037 | 50   | 1      | 2          | 5e-5          | 128       | 16.0       | False          | done    |
| ttt_adapters_llama3/ttt_adapters_llama3_70_epochs_1 | tba      | 70   | 1      | 2          | 5e-5          | 128       | 16.0       | False          | ongoing |






 ttt_adapters_llama3/ttt_adapters_llama3_nmax_10_batch_2_ep_1_lr_5e-5_rank_128_alpha_16.0/0bb8deee/