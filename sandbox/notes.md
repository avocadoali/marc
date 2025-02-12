
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

Adapters:
 ├── adapter_config_1.json
│   ├── adapter_config_2.json
│   ├── adapter_config_3.json
│   ├── adapter_config_4.json
│   ├── adapter_config_5.json
│   ├── adapter_model.bin_1
│   ├── adapter_model.bin_2
│   ├── adapter_model.bin_3
│   ├── adapter_model.bin_4
│   ├── adapter_model.bin_5


I want to run the prediction. Now each adapter directory has more adapter configs (e.g. adapter_config_1.json, adapter_config_2.json, etc.) and more adapter models (e.g. adapter_model.bin_1, adapter_model.bin_2, etc.). 
Edit the code such that I can run the prediction for each adapter config and adapter model.



/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/



[nguyen31@jwlogin24 ttt_adapters_llama3_nmax_1300_batch_2_ep_1_lr_5e-5_rank_128_alpha_16.0]$ tree -L 4
.
├── 00576224
│   ├── adapter_config.json
│   ├── adapter_model_10.bin
│   ├── adapter_model_20.bin
│   ├── adapter_model_2.bin
│   ├── adapter_model_40.bin
│   ├── adapter_model_4.bin
│   ├── adapter_model_60.bin
│   ├── adapter_model_6.bin
│   ├── adapter_model_80.bin
│   ├── adapter_model_8.bin
│   ├── config.json
│   ├── log_1738284345.txt
│   ├── td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl
│   └── td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl
├── 03560426
│   ├── adapter_config.json
│   ├── adapter_model_100.bin
│   ├── adapter_model_10.bin
│   ├── adapter_model_200.bin
│   ├── adapter_model_20.bin
│   ├── adapter_model_2.bin
│   ├── adapter_model_40.bin
│   ├── adapter_model_4.bin
│   ├── adapter_model_60.bin
│   ├── adapter_model_6.bin
│   ├── adapter_model_80.bin
│   ├── adapter_model_8.bin
│   ├── config.json
│   ├── log_1738284396.txt
│   ├── td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl
│   └── td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl
├── 0607ce86
│   ├── adapter_config.json
│   ├── adapter_model_10.bin
│   ├── adapter_model_20.bin
│   ├── adapter_model_2.bin
│   ├── adapter_model_40.bin
│   ├── adapter_model_4.bin
│   ├── adapter_model_60.bin
│   ├── adapter_model_6.bin
│   ├── adapter_model_8.bin
│   ├── config.json
│   ├── log_1738284838.txt
│   ├── td_False_ttd_False_ttdwa_False_ad_True_trd_False.jsonl
│   └── td_True_ttd_False_ttdwa_False_ad_True_trd_False.jsonl



give me a command tomove the adapters into the following directory structure:

adapters_10
- 00576224
  - adapter_config.json
  - adapter_model_10.bin
- 03560426
  - adapter_config.json
  - adapter_model_10.bin
- 0607ce86
  - adapter_config.json
  - adapter_model_10.bin

adapters_20
- 00576224
  - adapter_config.json
  - adapter_model_20.bin
- 03560426
  - adapter_config.json
  - adapter_model_20.bin
- 0607ce86
  - adapter_config.json
  - adapter_model_20.bin

adapters_30
- 00576224
  - adapter_config.json
  - adapter_model_30.bin
- 03560426
  - adapter_config.json
  - adapter_model_30.bin
- 0607ce86
  - adapter_config.json
  - adapter_model_30.bin



025-01-31 21:42:54,670 - torchtune.util

$ python debug_transformations.py --lora_config=configs/ttt/8B_lora_single_device.yaml --base_checkpoint_d
ir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628
ef6e0a75fef50264c91b142a --experiment_folder=transformation_test --data_file=arc-prize-2024/arc-agi_evaluation_challenges.json --batch_size=2 --o
ffset=0 --num_tasks=100 --Nmax=1500 --epochs=1 --lora_rank=128 --lora_alpha=16.0 --lora_to_output=False --new_format

