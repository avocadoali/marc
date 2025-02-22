import os
import matplotlib.pyplot as plt

# Directory containing the adapter directories
base_directory = 'experiments_thesis/2k_test_run'

# Dictionary to store the dataset sizes for each adapter
adapter_sizes = {}

# Iterate over each directory in the base directory
for directory in os.listdir(base_directory):
    if directory.startswith('adapters_json_'):
        # Extract the dataset size from the directory name
        try:
            dataset_size = int(directory.split('_')[-1])
            adapter_path = os.path.join(base_directory, directory)
            if os.path.isdir(adapter_path):
                # Iterate over each adapter in the directory
                for adapter in os.listdir(adapter_path):
                    adapter_sizes[adapter] = adapter_sizes.get(adapter, []) + [dataset_size]
        except ValueError:
            # Skip directories that don't end with a number
            continue

# Print the adapter sizes dictionary
# order the adapters by their natural order
sorted_adapters = sorted(adapter_sizes.keys())
# print(sorted_adapters)

# order the lists of dataset sizes by size
for adapter in sorted_adapters:
    adapter_sizes[adapter].sort()
print(adapter_sizes)

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Iterate over each adapter and plot its dataset sizes
for adapter in sorted_adapters:
    sizes = adapter_sizes[adapter]
    plt.scatter([adapter] * len(sizes), sizes, label=adapter)

# Add labels and title
plt.xlabel('Adapters')
plt.ylabel('Dataset Sizes')
plt.title('Scatter Plot of Dataset Sizes for Each Adapter')
plt.xticks(rotation=45, ha='right')  # Rotate adapter names for better readability
plt.tight_layout()

# save the plot
plt.savefig('plots/adapter_dataset_sizes.png')