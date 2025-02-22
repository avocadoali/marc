import os
import matplotlib.pyplot as plt

# Directory containing the adapter directories
base_directory = 'experiments_thesis/2k_test_run'

# Dictionary to store the dataset size for each adapter
adapter_sizes = {}

# Iterate over each directory in the base directory
for directory in os.listdir(base_directory):
    if directory.startswith('adapters_json_'):
        # Extract the dataset size from the directory name
        try:
            dataset_size = int(directory.split('_')[-1])
            adapter_sizes[directory] = dataset_size
        except ValueError:
            # Skip directories that don't end with a number
            continue

# Print the adapter sizes dictionary
print(adapter_sizes)

# Dictionary to store the number of lines for each adapter
adapter_data = {}

# Iterate over each adapter directory
for adapter in os.listdir(base_directory):
    adapter_path = os.path.join(base_directory, adapter)
    if os.path.isdir(adapter_path):
        # Find the td_False*.jsonl file
        for filename in os.listdir(adapter_path):
            if filename.startswith('td_False') and filename.endswith('.jsonl'):
                filepath = os.path.join(adapter_path, filename)
                with open(filepath, 'r') as file:
                    # Count the number of lines in the file
                    line_count = sum(1 for _ in file)
                    adapter_data[adapter] = line_count
                break  # Assuming only one td_False*.jsonl file per adapter

# Sort the adapters by their natural order
sorted_adapters = sorted(adapter_data.keys())

# Plotting
plt.figure(figsize=(12, 8))
plt.plot(sorted_adapters, [adapter_data[adapter] for adapter in sorted_adapters], marker='o')

plt.title('Number of Lines in td_False*.jsonl per Adapter')
plt.xlabel('Adapter')
plt.ylabel('Number of Lines')
plt.xticks(rotation=90)
plt.grid(True)

# Save the plot
plt.savefig('plots/adapter_line_counts.png')