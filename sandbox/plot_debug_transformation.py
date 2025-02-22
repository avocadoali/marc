import os
import json
import matplotlib.pyplot as plt

# Directory containing the JSON files
directory = 'stats/stats_debug_transformations'

# Dictionary to store the number of transformed examples for each file
file_data = {}


# filename_list = ['stats_baseline.json',  'stats_perm_2_redo_init_3.json', 'stats_perm_2_redo_init_3_init_4.json']
filename_list = ['baseline_250_permute_1_ep_1.json', 'baseline_1000_permute_1_20k.json', 'baseline_1000_permute_2_20k.json']
# filename_list = ['stats_baseline.json', 'stats_perm_2.json', 'stats_perm_2_redo_init_3.json']

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # only plot baseline and 4 
    # if filename == 'stats_baseline.json' or filename == 'stats_20k_5.json' or filename == 'stats.json':
    # if filename == 'stats_baseline.json' or filename == 'stats_perm_2.json' or filename:
    # if filename == 'stats_perm_2.json':
    # if filename.startswith('stats_') and filename.endswith('.json') and filename.startswith('baseline') or filename.startswith('4'):
    if filename in filename_list:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            data = json.load(file)
            # Extract num_transformed_examples from each entry if smaller than 500
            # transformed_examples = [value['num_transformed_examples'] for value in data.values() if value['num_transformed_examples'] < 500]

            # extract num_transformed_examples for each task
            transformed_examples = [value['num_transformed_examples'] for value in data.values()]

            file_data[filename] = transformed_examples

# Plotting
plt.figure(figsize=(12, 8))
for filename, transformed_examples in file_data.items():
    plt.scatter(range(len(transformed_examples)), transformed_examples, marker='o', label=filename)

plt.title('Number of Transformed Examples per File')
plt.xlabel('Adapters Index')
plt.ylabel('Number of Transformed Examples')
plt.grid(True)
plt.legend()



# save the plot
plt.savefig('plots/debug_transformation_tmp.png')

# # get the data for stats_5.json and print all the idx that are 0
# data = file_data['stats_5.json']
# for idx, value in enumerate(data):
#     if value == 0:
#         print(f'idx: {idx}')
