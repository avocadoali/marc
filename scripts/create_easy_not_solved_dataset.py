import json
import os
import shutil
import pandas as pd

def create_unsolved_easy_tasks_dataset(input_file, output_file, task_info_file):
    # Read the original dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Read the task info CSV
    df = pd.read_csv(task_info_file)
    
    # Filter for unsolved easy tasks
    unsolved_easy_tasks = df[(df['level'] == 'easy') & (df['correct'] == False)]['task_id'].tolist()
    
    # Create new dataset with only unsolved easy tasks
    unsolved_easy_data = {}
    for task_id in unsolved_easy_tasks:
        if task_id in data:
            unsolved_easy_data[task_id] = data[task_id]
        else:
            print(f"Warning: Task {task_id} not found in dataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the filtered dataset
    with open(output_file, 'w') as f:
        json.dump(unsolved_easy_data, f, indent=2)
    
    print(f"Created dataset with {len(unsolved_easy_data)} unsolved easy tasks")
    print(f"Output saved to: {output_file}")

def main():
    # Define input and output paths
    input_file = "arc-prize-2024/arc-agi_evaluation_challenges.json"  # Adjust path as needed
    output_file = "arc-prize-2024/arc-unsolved-easy-tasks.json"
    task_info_file = "ttt_output_complete/ttt_output_1000/task_info.csv"
    
    create_unsolved_easy_tasks_dataset(input_file, output_file, task_info_file)

if __name__ == "__main__":
    main()
