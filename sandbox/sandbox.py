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







22 marc]$ sinfo -p booster
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
booster      up   infinite     11   plnd jwb[0127,0193,0197-0198,0254,0548,0553,0697,1023,1126,1151]
booster      up   infinite     16 drain* jwb[0058,0262,0392,0418,0422,0479,0607,0646,0671,0699,0725,0748,0759,1065,1153,1189]
booster      up   infinite     13   comp jwb[0355,0361,0375,0490,0492,0506-0507,0510,0512,0829,0833,1018,1088]
booster      up   infinite      6  drain jwb[0245,0664,0824,0840,1025,1027]
booster      up   infinite    861  alloc jwb[0002-0012,0022-0024,0026-0029,0031-0032,0034-0044,0054-0057,0059-0064,0066-0076,0086-0093,0095,0098-0108,0118-0126,0128,0130-0140,0150-0172,0181-0192,0195-0196,0200-0204,0213-0236,0246-0253,0255-0261,0263-0264,0267-0268,0277-0300,0309-0332,0341-0354,0356-0360,0362-0364,0373-0374,0376-0391,0393-0396,0405-0417,0419-0421,0423-0428,0437-0460,0469-0478,0480-0489,0491,0501-0505,0508-0509,0511,0513-0524,0533-0547,0549-0552,0554-0556,0565-0572,0574-0575,0577-0588,0597-0606,0608-0620,0629-0645,0647-0652,0661-0663,0665-0669,0672-0684,0693-0696,0698,0700-0716,0726-0747,0757-0758,0760-0780,0789-0812,0821-0823,0825-0828,0830-0832,0834-0839,0841-0844,0853-0876,0885-0894,0896-0908,0917-0940,0949-0956,0958-0972,0981,0983,0985-1004,1013-1017,1019-1022,1024,1026,1028-1036,1045-1064,1066-1068,1077-1087,1089-1100,1109-1125,1127-1132,1141-1142,1144-1145,1147-1150,1152,1154-1164,1173-1188,1190-1196,1205-1220,1222-1228,1237-1242,1244-1248]
booster      up   infinite     19   idle jwb[0025,0030,0094,0096,0194,0199,0265-0266,0573,0576,0670,0895,0957,0982,0984,1143,1146,1221,1243]