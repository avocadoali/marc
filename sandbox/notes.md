I want to run a torchtune lora finetune on multiple gpus. However I dont have internet access on the compute node. What master address and port do I need to set?


tune run --nproc_per_node 4 lora_finetune_distributed --config configs/ttt/8B_lora_multi.yaml checkpointer.checkpoint_dir=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/ > tee logs_ttt/torchtune_distributed_test.log




Dear Tech Support,

I'm trying to run a torchtune lora finetune on multiple gpus on this [repo](https://github.com/ekinakyurek/marc?tab=readme-ov-file). However, it doesn't work for me. 

 
I then tried to setup the official [torchtune repo](https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama3/8B_lora.yaml) and run the example there. However, I get the following error:

**insert screenshot here**

[W117 15:26:10.029795075 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29500 (errno: 97 - Address family not supported by protocol).


I suppose it has something to do with network restricitons on the compute node. Has anyone else run into this issue or tried to setup some distributed torchtune finetuning?

Thanks in advance for the help!

Greetings,
Alfred


PS:

I have copied the output here in this file:

/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/torchtune/torch_tune_pasted.log


You should be able to recreate the error with the following command:

``` bash 
# cd to /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/torchtune
$ source ../marc/sc_venv_arc/activate.sh 
$ tune run --nproc_per_node 2 lora_finetune_distributed --config  recipes/configs/llama3_1/8B_lora.yaml 
``` 
 






INFO 01-10 18:08:47 llm_engine.py:184] Initializing an LLM engine (v0.5.4) with config: model='/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', speculative_config=None, tokenizer='/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a, use_v2_block_manager=False, enable_prefix_caching=False)



WARNING 01-10 18:29:08 tokenizer.py:152] No tokenizer found in /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/67c52801, using base model tokenizer instead. (Exception: Can't load tokenizer for '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/67c52801'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/67c52801' is the correct path to a directory containing all relevant files for a LlamaTokenizerFast tokenizer.)

DEBUG 01-10 18:29:08 models.py:35] Removing adapter int id: 240



ARNING 01-10 18:31:21 tokenizer.py:152] No tokenizer found in /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/506d28a5, using base model tokenizer instead. (Exception: Can't load tokenizer for '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/506d28a5'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/506d28a5' is the correct path to a directory containing all relevant files for a LlamaTokenizerFast tokenizer.)
WARNING 01-10 18:31:22 tokenizer.py:152] No tokenizer found in /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/73c3b0d8, using base model tokenizer instead. (Exception: Can't load tokenizer for '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/73c3b0d8'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/73c3b0d8' is the correct path to a directory containing all relevant files for a LlamaTokenizerFast tokenizer.)
WARNING 01-10 18:31:23 tokenizer.py:152] No tokenizer found in /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/ea9794b1, using base model tokenizer instead. (Exception: Can't load tokenizer for '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/ea9794b1'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/ea9794b1' is the correct path to a directory containing all relevant files for a LlamaTokenizerFast tokenizer.)

INFO 01-10 18:31:23 metrics.py:406] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 164.4 tokens/s, Running: 10 reqs, Swapped: 0 reqs, Pending: 4550 reqs, GPU KV cache usage: 27.6%, CPU KV cache usage: 0.0%.




WARNING 01-10 18:31:24 tokenizer.py:152] No tokenizer found in /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-finetuned-llama3/snapshots/0bfc91056465763e61d86bb047955364a82eaee2/b7cb93ac, using base model tokenizer instead. (Exception: Can't load tokenizer for '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-lora-adapters-8B-fine









Code to create dataset to train MLP + estimate how much compute OR just ask for the dataset if too much trouble
Code to recreate MLP based difficulty estimation and LoRA based difficulty estimation.
Code for the maths and/or code experiments to serve as a baseline to the log prob idea (I would probably switch to a smaller model due to lack of compute)
Ideally: Code to run the experiment/method on MATH or GSM8K. Is it possible to switch out models using the code? E.g. use Llama 1B/3B for initial experiments?





















1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,507 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
2025-01-17 17:17:56,507 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
2025-01-17 17:17:56,507 - __main__ - DEBUG - Config: {'tokenizer': {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}, 'model': {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}, 'dataset': {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'data/dummy/', 'train_on_input': False, 'unmask_outputs': True}, 'checkpointer': {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ali--marc-lora-adapters-8B-finetuned-llama3', 'model_type': 'LLAMA3'}, 'resume_from_checkpoint': False, 'save_adapter_weights_only': True, 'seed': 0, 'shuffle': True, 'batch_size': 1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,507 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
2025-01-17 17:17:56,523 - __main__ - DEBUG - Starting test time training: 1737130676.5232255
2025-01-17 17:17:56,523 - __main__ - DEBUG - Available GPUs: 4
2025-01-17 17:17:56,644 - __main__ - DEBUG - Number of train tasks: 2
2025-01-17 17:17:56,669 - __main__ - DEBUG - Config: {'tokenizer': {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}, 'model': {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}, 'dataset': {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'data/dummy/', 'train_on_input': False, 'unmask_outputs': True}, 'checkpointer': {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ali--marc-lora-adapters-8B-finetuned-llama3', 'model_type': 'LLAMA3'}, 'resume_from_checkpoint': False, 'save_adapter_weights_only': True, 'seed': 0, 'shuffle': True, 'batch_size': 1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,669 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
2025-01-17 17:18:19,933 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
Writing logs to experiments/lora/log_1737130699.txt
2025-01-17 17:18:19,954 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
2025-01-17 17:18:19,954 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
2025-01-17 17:18:19,960 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
Writing logs to experiments/lora/log_1737130699.txt
Writing logs to experiments/lora/log_1737130699.txt
Writing logs to experiments/lora/log_1737130699.txt
2025-01-17 17:18:21,933 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
2025-01-17 17:18:21,933 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
2025-01-17 17:18:21,934 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
2025-01-17 17:18:21,935 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
[W117 17:18:22.588115933 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
[W117 17:18:22.588248374 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
[W117 17:18:22.588413365 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
[W117 17:18:22.588527926 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
2025-01-17 17:19:44,332 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,333 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,333 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,334 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,334 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,335 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,337 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,338 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,630 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,631 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,632 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,632 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,633 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,633 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,635 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,636 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 139886014180864 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 140636851432144 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,764 - filelock - DEBUG - Lock 139886014180864 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,765 - filelock - DEBUG - Lock 140636851432144 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,765 - filelock - DEBUG - Lock 140692529212784 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,765 - filelock - DEBUG - Lock 140671340407840 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,766 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,767 - filelock - DEBUG - Attempting to release lock 139886014180864 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,767 - filelock - DEBUG - Lock 139886014180864 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Attempting to acquire lock 140636851432144 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Attempting to acquire lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Lock 140636851432144 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Lock 140692529212784 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,815 - filelock - DEBUG - Lock 140671340407840 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,815 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,816 - filelock - DEBUG - Attempting to release lock 140636851432144 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,816 - filelock - DEBUG - Lock 140636851432144 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,865 - filelock - DEBUG - Attempting to acquire lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,865 - filelock - DEBUG - Lock 140692529212784 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,865 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,866 - filelock - DEBUG - Lock 140671340407840 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,866 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,867 - filelock - DEBUG - Attempting to release lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,867 - filelock - DEBUG - Lock 140692529212784 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,916 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,916 - filelock - DEBUG - Lock 140671340407840 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,917 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,918 - filelock - DEBUG - Attempting to release lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,918 - filelock - DEBUG - Lock 140671340407840 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:45,080 - filelock - DEBUG - Attempting to acquire lock 139886014259088 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,080 - filelock - DEBUG - Attempting to acquire lock 140636851347536 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,080 - filelock - DEBUG - Attempting to acquire lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,081 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 139886014259088 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 140636851347536 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 140671340470112 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 140692529307248 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,083 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,083 - filelock - DEBUG - Attempting to release lock 139886014259088 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 139886014259088 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Attempting to acquire lock 140636851347536 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Attempting to acquire lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Lock 140636851347536 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Lock 140671340470112 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,133 - filelock - DEBUG - Lock 140692529307248 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,134 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,134 - filelock - DEBUG - Attempting to release lock 140636851347536 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,134 - filelock - DEBUG - Lock 140636851347536 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Attempting to acquire lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Lock 140671340470112 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Lock 140692529307248 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,184 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,185 - filelock - DEBUG - Attempting to release lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,185 - filelock - DEBUG - Lock 140671340470112 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,234 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,235 - filelock - DEBUG - Lock 140692529307248 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,235 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,236 - filelock - DEBUG - Attempting to release lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,236 - filelock - DEBUG - Lock 140692529307248 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,483 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,483 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,483 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,484 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
2025-01-17 17:19:45,486 - torchtune.utils._logging - WARNING -  Profiling disabled.
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Train data size:  250
====CONFIG FOR 009d5c81====
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,376 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
Train data size:  250
====CONFIG FOR 009d5c81====
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,379 - __main__ - DEBUG - Done finished training: 110.85595560073853
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,379 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,380 - __main__ - DEBUG - Done finished training: 110.99803471565247
Train data size:  250
====CONFIG FOR 009d5c81====
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,388 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,389 - __main__ - DEBUG - Done finished training: 111.00691723823547
Train data size:  250
====CONFIG FOR 009d5c81====
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,394 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,395 - __main__ - DEBUG - Done finished training: 111.01285886764526
(sc_venv_arc) [nguyen31@jwb0256 marc]$ 
































sc_venv_arc) [nguyen31@jwb0256 marc]$ ./run_ttt.sh 
Master address: localhost
Master port: 29601
World size: 4
Rank: 0
W0117 17:13:58.381139 19104 torch/distributed/run.py:793] 
W0117 17:13:58.381139 19104 torch/distributed/run.py:793] *****************************************
W0117 17:13:58.381139 19104 torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0117 17:13:58.381139 19104 torch/distributed/run.py:793] *****************************************
Traceback (most recent call last):
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 260, in launch_agent
    result = agent.run()
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 137, in wrapper
    result = f(*args, **kwargs)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 711, in run
    result = self._invoke_run(role)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 864, in _invoke_run
    self._initialize_workers(self._worker_group)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 137, in wrapper
    result = f(*args, **kwargs)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 683, in _initialize_workers
    self._rendezvous(worker_group)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/metrics/api.py", line 137, in wrapper
    result = f(*args, **kwargs)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/agent/server/api.py", line 500, in _rendezvous
    rdzv_info = spec.rdzv_handler.next_rendezvous()
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py", line 67, in next_rendezvous
    self._store = TCPStore(  # type: ignore[call-arg]
RuntimeError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: 0, code: -98, name: EADDRINUSE, message: address already in use
(sc_venv_arc) [nguyen31@jwb0256 marc]$ jobs
[1]-  Stopped                 ./run_ttt.sh
[2]+  Stopped                 ./run_ttt.sh
(sc_venv_arc) [nguyen31@jwb0256 marc]$ kill $(jobs -p)
(sc_venv_arc) [nguyen31@jwb0256 marc]$ jobs
[1]-  Stopped                 ./run_ttt.sh
[2]+  Stopped                 ./run_ttt.sh
(sc_venv_arc) [nguyen31@jwb0256 marc]$ kill %1
[1]-  Terminated              ./run_ttt.sh
(sc_venv_arc) [nguyen31@jwb0256 marc]$ kill %2
[2]+  Terminated              ./run_ttt.sh
(sc_venv_arc) [nguyen31@jwb0256 marc]$ jobs
(sc_venv_arc) [nguyen31@jwb0256 marc]$ ^C
(sc_venv_arc) [nguyen31@jwb0256 marc]$ jobs^C
(sc_venv_arc) [nguyen31@jwb0256 marc]$ ./run_ttt.sh 
Master address: localhost
Master port: 29601
World size: 4
Rank: 0
W0117 17:15:30.320294 20440 torch/distributed/run.py:793] 
W0117 17:15:30.320294 20440 torch/distributed/run.py:793] *****************************************
W0117 17:15:30.320294 20440 torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0117 17:15:30.320294 20440 torch/distributed/run.py:793] *****************************************
[W117 17:15:30.872966799 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29500 (errno: 97 - Address family not supported by protocol).
/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/python3: can't open file '/p/project1/hai_hreplearn/nguyen31/marc/python': [Errno 2] No such file or directory
/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/python3: can't open file '/p/project1/hai_hreplearn/nguyen31/marc/python': [Errno 2] No such file or directory
/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/python3: can't open file '/p/project1/hai_hreplearn/nguyen31/marc/python': [Errno 2] No such file or directory
/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/python3: can't open file '/p/project1/hai_hreplearn/nguyen31/marc/python': [Errno 2] No such file or directory
E0117 17:15:30.439273 20440 torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 2) local_rank: 0 (pid: 20555) of binary: /p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/python3
Traceback (most recent call last):
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/run.py", line 919, in main
    run(args)
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/run.py", line 910, in run
    elastic_launch(
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/p/project1/hai_hreplearn/nguyen31/marc/sc_venv_arc/venv/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
python FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2025-01-17_17:15:30
  host      : jwb0256.juwels
  rank      : 1 (local_rank: 1)
  exitcode  : 2 (pid: 20556)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-01-17_17:15:30
  host      : jwb0256.juwels
  rank      : 2 (local_rank: 2)
  exitcode  : 2 (pid: 20557)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-01-17_17:15:30
  host      : jwb0256.juwels
  rank      : 3 (local_rank: 3)
  exitcode  : 2 (pid: 20558)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-01-17_17:15:30
  host      : jwb0256.juwels
  rank      : 0 (local_rank: 0)
  exitcode  : 2 (pid: 20555)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
(sc_venv_arc) [nguyen31@jwb0256 marc]$ ./run_ttt.sh 
W0117 17:17:43.885468 22504 torch/distributed/run.py:793] 
W0117 17:17:43.885468 22504 torch/distributed/run.py:793] *****************************************
W0117 17:17:43.885468 22504 torch/distributed/run.py:793] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0117 17:17:43.885468 22504 torch/distributed/run.py:793] *****************************************
[W117 17:17:43.438228810 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
UNMASK TOKENS IN CURRENT MODE
UNMASK TOKENS IN CURRENT MODE
UNMASK TOKENS IN CURRENT MODE
UNMASK TOKENS IN CURRENT MODE
2025-01-17 17:17:56,382 - __main__ - DEBUG - Starting test time training: 1737130676.382129
2025-01-17 17:17:56,382 - __main__ - DEBUG - Available GPUs: 4
2025-01-17 17:17:56,382 - __main__ - DEBUG - Starting test time training: 1737130676.3823955
2025-01-17 17:17:56,382 - __main__ - DEBUG - Available GPUs: 4
2025-01-17 17:17:56,382 - __main__ - DEBUG - Starting test time training: 1737130676.3826883
2025-01-17 17:17:56,382 - __main__ - DEBUG - Available GPUs: 4
2025-01-17 17:17:56,483 - __main__ - DEBUG - Number of train tasks: 2
2025-01-17 17:17:56,483 - __main__ - DEBUG - Number of train tasks: 2
2025-01-17 17:17:56,484 - __main__ - DEBUG - Number of train tasks: 2
2025-01-17 17:17:56,507 - __main__ - DEBUG - Config: {'tokenizer': {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}, 'model': {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}, 'dataset': {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'data/dummy/', 'train_on_input': False, 'unmask_outputs': True}, 'checkpointer': {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ali--marc-lora-adapters-8B-finetuned-llama3', 'model_type': 'LLAMA3'}, 'resume_from_checkpoint': False, 'save_adapter_weights_only': True, 'seed': 0, 'shuffle': True, 'batch_size': 1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,507 - __main__ - DEBUG - Config: {'tokenizer': {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}, 'model': {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}, 'dataset': {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'data/dummy/', 'train_on_input': False, 'unmask_outputs': True}, 'checkpointer': {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ali--marc-lora-adapters-8B-finetuned-llama3', 'model_type': 'LLAMA3'}, 'resume_from_checkpoint': False, 'save_adapter_weights_only': True, 'seed': 0, 'shuffle': True, 'batch_size': 1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,507 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
2025-01-17 17:17:56,507 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
2025-01-17 17:17:56,507 - __main__ - DEBUG - Config: {'tokenizer': {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}, 'model': {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}, 'dataset': {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'data/dummy/', 'train_on_input': False, 'unmask_outputs': True}, 'checkpointer': {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ali--marc-lora-adapters-8B-finetuned-llama3', 'model_type': 'LLAMA3'}, 'resume_from_checkpoint': False, 'save_adapter_weights_only': True, 'seed': 0, 'shuffle': True, 'batch_size': 1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,507 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
2025-01-17 17:17:56,523 - __main__ - DEBUG - Starting test time training: 1737130676.5232255
2025-01-17 17:17:56,523 - __main__ - DEBUG - Available GPUs: 4
2025-01-17 17:17:56,644 - __main__ - DEBUG - Number of train tasks: 2
2025-01-17 17:17:56,669 - __main__ - DEBUG - Config: {'tokenizer': {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}, 'model': {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}, 'dataset': {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'data/dummy/', 'train_on_input': False, 'unmask_outputs': True}, 'checkpointer': {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ali--marc-lora-adapters-8B-finetuned-llama3', 'model_type': 'LLAMA3'}, 'resume_from_checkpoint': False, 'save_adapter_weights_only': True, 'seed': 0, 'shuffle': True, 'batch_size': 1, 'optimizer': {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}, 'lr_scheduler': {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 5}, 'loss': {'_component_': 'torch.nn.CrossEntropyLoss'}, 'epochs': 2, 'max_steps_per_epoch': None, 'gradient_accumulation_steps': 1, 'compile': False, 'output_dir': 'experiments/lora', 'metric_logger': {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}, 'log_every_n_steps': 1, 'log_peak_memory_stats': False, 'device': 'cuda', 'dtype': 'bf16', 'enable_activation_checkpointing': True, 'enable_activation_offloading': False, 'profiler': {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}}
2025-01-17 17:17:56,669 - __main__ - DEBUG - Tokenizer path: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
Augmenters to apply:  [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1), Reflect(0, reverse=True), Reflect(1, reverse=True), Reflect(0, reverse=False), Reflect(1, reverse=False), RandomTranslateXY(), Transpose(), IncreaseResolution(2), IncreaseHeight(2), IncreaseWidth(2), Chain([Rotate(90), IncreaseResolution(2)]), Chain([Rotate(270), IncreaseResolution(2)]), Chain([Rotate(180), IncreaseResolution(2)]), Chain([Flip(0), IncreaseResolution(2)]), Chain([Flip(1), IncreaseResolution(2)]), Chain([Transpose(), IncreaseResolution(2)]), Repeat(0, 2), Repeat(1, 2), Repeat(2, 2)] len:  23
2025-01-17 17:18:19,933 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
Writing logs to experiments/lora/log_1737130699.txt
2025-01-17 17:18:19,954 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
2025-01-17 17:18:19,954 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
2025-01-17 17:18:19,960 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
Writing logs to experiments/lora/log_1737130699.txt
Writing logs to experiments/lora/log_1737130699.txt
Writing logs to experiments/lora/log_1737130699.txt
2025-01-17 17:18:21,933 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
2025-01-17 17:18:21,933 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
2025-01-17 17:18:21,934 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
2025-01-17 17:18:21,935 - torchtune.utils._logging - INFO - FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ...
[W117 17:18:22.588115933 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
[W117 17:18:22.588248374 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
[W117 17:18:22.588413365 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
[W117 17:18:22.588527926 socket.cpp:758] [c10d] The client socket cannot be initialized to connect to [localhost.localdomain]:29632 (errno: 97 - Address family not supported by protocol).
2025-01-17 17:19:44,332 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,333 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,333 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,334 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,334 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,335 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,337 - torchtune.utils._logging - INFO - Instantiating model and loading checkpoint took 82.40 secs
2025-01-17 17:19:44,338 - torchtune.utils._logging - INFO - Memory stats after model init:
        GPU peak memory allocation: 4.98 GiB
        GPU peak memory reserved: 5.12 GiB
        GPU peak memory active: 4.98 GiB
2025-01-17 17:19:44,630 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,631 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,632 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,632 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,633 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,633 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,635 - torchtune.utils._logging - INFO - Optimizer is initialized.
2025-01-17 17:19:44,636 - torchtune.utils._logging - INFO - Loss is initialized.
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 139886014180864 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 140636851432144 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,657 - filelock - DEBUG - Attempting to acquire lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,764 - filelock - DEBUG - Lock 139886014180864 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,765 - filelock - DEBUG - Lock 140636851432144 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,765 - filelock - DEBUG - Lock 140692529212784 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,765 - filelock - DEBUG - Lock 140671340407840 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,766 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,767 - filelock - DEBUG - Attempting to release lock 139886014180864 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,767 - filelock - DEBUG - Lock 139886014180864 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Attempting to acquire lock 140636851432144 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Attempting to acquire lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Lock 140636851432144 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,815 - filelock - DEBUG - Lock 140692529212784 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,815 - filelock - DEBUG - Lock 140671340407840 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,815 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,816 - filelock - DEBUG - Attempting to release lock 140636851432144 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,816 - filelock - DEBUG - Lock 140636851432144 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,865 - filelock - DEBUG - Attempting to acquire lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,865 - filelock - DEBUG - Lock 140692529212784 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,865 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,866 - filelock - DEBUG - Lock 140671340407840 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock, waiting 0.05 seconds ...
2025-01-17 17:19:44,866 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,867 - filelock - DEBUG - Attempting to release lock 140692529212784 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,867 - filelock - DEBUG - Lock 140692529212784 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,916 - filelock - DEBUG - Attempting to acquire lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,916 - filelock - DEBUG - Lock 140671340407840 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,917 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:44,918 - filelock - DEBUG - Attempting to release lock 140671340407840 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:44,918 - filelock - DEBUG - Lock 140671340407840 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/_p_home_jusers_nguyen31_juwels_arc-challenge_nguyen31_huggingface_datasets_dummy_default-a525bd183ff89302_0.0.0_76ccb0a9dd389f5d.lock
2025-01-17 17:19:45,080 - filelock - DEBUG - Attempting to acquire lock 139886014259088 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,080 - filelock - DEBUG - Attempting to acquire lock 140636851347536 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,080 - filelock - DEBUG - Attempting to acquire lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,081 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 139886014259088 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 140636851347536 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 140671340470112 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 140692529307248 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,083 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,083 - filelock - DEBUG - Attempting to release lock 139886014259088 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,083 - filelock - DEBUG - Lock 139886014259088 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Attempting to acquire lock 140636851347536 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Attempting to acquire lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Lock 140636851347536 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,133 - filelock - DEBUG - Lock 140671340470112 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,133 - filelock - DEBUG - Lock 140692529307248 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,134 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,134 - filelock - DEBUG - Attempting to release lock 140636851347536 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,134 - filelock - DEBUG - Lock 140636851347536 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Attempting to acquire lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Lock 140671340470112 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,184 - filelock - DEBUG - Lock 140692529307248 not acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock, waiting 0.05 seconds ...
2025-01-17 17:19:45,184 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,185 - filelock - DEBUG - Attempting to release lock 140671340470112 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,185 - filelock - DEBUG - Lock 140671340470112 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,234 - filelock - DEBUG - Attempting to acquire lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,235 - filelock - DEBUG - Lock 140692529307248 acquired on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,235 - fsspec.local - DEBUG - open file: /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d/dataset_info.json
2025-01-17 17:19:45,236 - filelock - DEBUG - Attempting to release lock 140692529307248 on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,236 - filelock - DEBUG - Lock 140692529307248 released on /p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/datasets/dummy/default-a525bd183ff89302/0.0.0/76ccb0a9dd389f5d_builder.lock
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,482 - torchtune.utils._logging - INFO - Dataset and Sampler are initialized.
2025-01-17 17:19:45,483 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,483 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,483 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,484 - torchtune.utils._logging - INFO - Learning rate scheduler is initialized.
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
2025-01-17 17:19:45,486 - torchtune.utils._logging - WARNING -  Profiling disabled.
2025-01-17 17:19:45,486 - torchtune.utils._logging - INFO -  Profiler config after instantiation: {'enabled': False}
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Trying task 00576224
Adapter for 00576224 already exists, skipping
Trying task 009d5c81
Train data size:  250
====CONFIG FOR 009d5c81====
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,376 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
Train data size:  250
====CONFIG FOR 009d5c81====
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,379 - __main__ - DEBUG - Done finished training: 110.85595560073853
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,379 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,380 - __main__ - DEBUG - Done finished training: 110.99803471565247
Train data size:  250
====CONFIG FOR 009d5c81====
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,388 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,389 - __main__ - DEBUG - Done finished training: 111.00691723823547
Train data size:  250
====CONFIG FOR 009d5c81====
tokenizer: {'_component_': 'torchtune.models.llama3.llama3_tokenizer', 'path': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a/tokenizer.model', 'max_seq_len': None}
model: {'_component_': 'torchtune.models.llama3.lora_llama3_8b', 'lora_attn_modules': ['q_proj', 'v_proj'], 'apply_lora_to_mlp': True, 'apply_lora_to_output': True, 'lora_rank': 128, 'lora_alpha': 16.0, 'lora_dropout': 0.0}
dataset: {'_component_': 'torchtune.datasets.arc_dataset', 'source': 'ttt_adapters/009d5c81', 'train_on_input': False, 'unmask_outputs': True}
checkpointer: {'_component_': 'torchtune.training.FullModelHFCheckpointer', 'checkpoint_dir': '/p/home/jusers/nguyen31/juwels/arc-challenge/nguyen31/huggingface/hub/models--ekinakyurek--marc-8B-finetuned-llama3/snapshots/c2b6b30b45e87628ef6e0a75fef50264c91b142a', 'checkpoint_files': ['pytorch_model-0001-of-0004.bin', 'pytorch_model-0002-of-0004.bin', 'pytorch_model-0003-of-0004.bin', 'pytorch_model-0004-of-0004.bin'], 'recipe_checkpoint': None, 'output_dir': 'ttt_adapters/009d5c81', 'model_type': 'LLAMA3'}
resume_from_checkpoint: False
save_adapter_weights_only: True
seed: 0
shuffle: True
batch_size: 1
optimizer: {'_component_': 'torch.optim.AdamW', 'fused': True, 'weight_decay': 0.01, 'lr': 0.0001}
lr_scheduler: {'_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup', 'num_warmup_steps': 50}
loss: {'_component_': 'torch.nn.CrossEntropyLoss'}
epochs: 2
max_steps_per_epoch: None
gradient_accumulation_steps: 1
compile: False
output_dir: ttt_adapters/009d5c81
metric_logger: {'_component_': 'torchtune.training.metric_logging.DiskLogger', 'log_dir': '${output_dir}'}
log_every_n_steps: 1
log_peak_memory_stats: False
device: cuda
dtype: bf16
enable_activation_checkpointing: True
enable_activation_offloading: False
profiler: {'_component_': 'torchtune.training.setup_torch_profiler', 'enabled': False, 'output_dir': '${output_dir}/profiling_outputs', 'cpu': True, 'cuda': True, 'profile_memory': False, 'with_stack': False, 'record_shapes': True, 'with_flops': False, 'wait_steps': 5, 'warmup_steps': 5, 'active_steps': 2, 'num_cycles': 1}
=============================
2025-01-17 17:19:47,394 - torchtune.utils._logging - DEBUG - Setting manual seed to local seed 0. Local seed is seed + rank = 0 + 0
LoRAFinetuneRecipeDistributed.setup() got an unexpected keyword argument 'model'
Error training for  009d5c81
2025-01-17 17:19:47,395 - __main__ - DEBUG - Done finished training: 111.01285886764526
(sc_venv_arc) [nguyen31@jwb0256 marc]$ 