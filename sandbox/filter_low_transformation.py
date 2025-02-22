import json

# Load the JSON data
with open('stats/stats_debug_transformations/stats.json', 'r') as file:
    data = json.load(file)

# Filter IDs and their indices
filtered_ids_with_idx = [(idx, key) for idx, (key, value) in enumerate(data.items()) if value['num_transformed_examples'] < 500]

# Print the filtered IDs with their indices
for idx, id in filtered_ids_with_idx:
    print(f"Index: {idx}, ID: {id}")

