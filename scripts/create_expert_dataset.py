import json
import os
import shutil

# List of hard task IDs
hard_tasks = [
    "e99362f0", "1acc24af", "f9a67cb5", "ad7e01d0", "ea9794b1",
    "58e15b12", "891232d6", "5833af48", "4ff4c9da", "5b692c0f",
    "e2092e0c", "0934a4d8", "47996f11", "0c9aba6e", "34b99a2b",
    "1c56ad9f", "e6de6e8f", "fea12743", "31d5ba1a", "79fb03f4",
    "8719f442", "a8610ef7", "b4a43f3b"
]

def create_hard_tasks_dataset(input_file, output_file):
    # Read the original dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create new dataset with only hard tasks
    hard_tasks_data = {}
    for task_id in hard_tasks:
        if task_id in data:
            hard_tasks_data[task_id] = data[task_id]
        else:
            print(f"Warning: Task {task_id} not found in dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the filtered dataset
    with open(output_file, 'w') as f:
        json.dump(hard_tasks_data, f, indent=2)
    
    print(f"Created dataset with {len(hard_tasks_data)} hard tasks")
    print(f"Output saved to: {output_file}")

def main():
    # Define input and output paths
    input_file = "arc-prize-2024/arc-agi_evaluation_challenges.json"  # Adjust path as needed
    output_file = "arc-prize-2024/arc-hard-tasks.json"
    
    create_hard_tasks_dataset(input_file, output_file)

if __name__ == "__main__":
    main()


2