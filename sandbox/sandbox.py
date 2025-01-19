import json
import os
import torch

# # You can run the main script
# torchrun --nproc_per_node 4 python test_time_train_multi.py --lora_config=$lora_config_file \
# list all cuda devices

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set this before importing torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Set this before importing torch



print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"CUDA available: {torch.cuda.is_available()}")
# list all cuda devices
print(f"Available GPUs: {torch.cuda.device_count()}")
try:
    print(f"Available GPUs: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error: {e}")

# # Read the submission file
# with open('ttt_output/submission.json', 'r') as f:
#     submission_data = json.load(f)

# # Read the all_predictions file
# with open('ttt_output/all_predictions.json', 'r') as f:
#     predictions_data = json.load(f)

# # Get stats for submission.json
# submission_count = len(submission_data)
# attempts_per_entry = [len(entry) for entry in submission_data.values()]
# total_attempts = sum(attempts_per_entry)

# # Get stats for all_predictions.json
# predictions_count = len(predictions_data)

# print("=== Submission.json Stats ===")
# print(f"Number of unique IDs: {submission_count}")
# print(f"Total number of attempts: {total_attempts}")
# print(f"Average attempts per ID: {total_attempts/submission_count:.2f}")

# print("\n=== All_predictions.json Stats ===")
# print(f"Number of entries: {predictions_count}")
