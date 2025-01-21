import json
import os
# import torch

# # # You can run the main script
# # torchrun --nproc_per_node 4 python test_time_train_multi.py --lora_config=$lora_config_file \
# # list all cuda devices

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Set this before importing torch

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Set this before importing torch



# print(f"Available GPUs: {torch.cuda.device_count()}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# # list all cuda devices
# print(f"Available GPUs: {torch.cuda.device_count()}")
# try:
#     print(f"Available GPUs: {torch.cuda.get_device_name(0)}")
# except Exception as e:
#     print(f"Error: {e}")

# # # Read the submission file
# # with open('ttt_output/submission.json', 'r') as f:
# #     submission_data = json.load(f)

# # # Read the all_predictions file
# # with open('ttt_output/all_predictions.json', 'r') as f:
# #     predictions_data = json.load(f)

# # # Get stats for submission.json
# # submission_count = len(submission_data)
# # attempts_per_entry = [len(entry) for entry in submission_data.values()]
# # total_attempts = sum(attempts_per_entry)

# # # Get stats for all_predictions.json
# # predictions_count = len(predictions_data)

# # print("=== Submission.json Stats ===")
# # print(f"Number of unique IDs: {submission_count}")
# # print(f"Total number of attempts: {total_attempts}")
# # print(f"Average attempts per ID: {total_attempts/submission_count:.2f}")

# # print("\n=== All_predictions.json Stats ===")
# # print(f"Number of entries: {predictions_count}")



# mock data
Nmax = 500
num_saved_adapters = 10
time_taken = 1000
arc_test_tasks = 1000
average_time_per_adapter_hours = 1
average_time_per_adapter_minutes = 1
average_time_per_adapter_seconds = 1
time_taken_hours = 1
time_taken_minutes = 1
time_taken_seconds = 1
arc_test_tasks = [1,2,3,4,5,6,7,8,9,10]


# import json
# import os

# stats = {
#     "Max Training Size": Nmax,
#     "Actual Duration": f"{time_taken_hours}:{time_taken_minutes}:{time_taken_seconds}",
#     "avg time per adapter": f"{average_time_per_adapter_hours}:{average_time_per_adapter_minutes}:{average_time_per_adapter_seconds}",
#     "#Created Adapters": num_saved_adapters,
#     "#To be created Adapters": len(arc_test_tasks),
# }

# # Try to read existing data or create new list
# if os.path.exists(f"stats/ttt_stats.json"):
#     with open(f"stats/ttt_stats.json", "r") as f:
#         try:
#             existing_stats = json.load(f)
#         except json.JSONDecodeError:
#             existing_stats = []
# else:
#     existing_stats = []

# # Ensure existing_stats is a list
# if not isinstance(existing_stats, list):
#     existing_stats = [existing_stats]

# # Append new stats
# existing_stats.append(stats)

# # Write back the complete list
# with open(f"stats/ttt_stats.json", "w") as f:
#     json.dump(existing_stats, f, indent=2)



#mock data
lora_checkpoints_folder = "ttt_adapters_50"
time_taken_hours = 1
time_taken_minutes = 1
time_taken_seconds = 1
average_time_per_adapter_hours = 1
average_time_per_adapter_minutes = 1
average_time_per_adapter_seconds = 1
corrects = 1
total = 1

stats = {
    "Max Training Size": lora_checkpoints_folder,
    "Actual Duration": f"{time_taken_hours}:{time_taken_minutes}:{time_taken_seconds}",
    "avg time per adapter": f"{average_time_per_adapter_hours}:{average_time_per_adapter_minutes}:{average_time_per_adapter_seconds}",
    "correct prediction": corrects,
    "total prediction": total,
}

# Try to read existing data or create new list
if os.path.exists(f"stats/predict_stats.json"):
    with open(f"stats/predict_stats.json", "r") as f:
        try:
            existing_stats = json.load(f)
        except json.JSONDecodeError:
            existing_stats = []
else:
    existing_stats = []

# Ensure existing_stats is a list
if not isinstance(existing_stats, list):
    existing_stats = [existing_stats]

# Append new stats
existing_stats.append(stats)

# Write back the complete list
with open(f"stats/predict_stats.json", "w") as f:
    json.dump(existing_stats, f, indent=2)




