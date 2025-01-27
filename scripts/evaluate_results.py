import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Function to extract model size from directory name
def get_model_size(dirname):
    # Extract number from ttt_output_X
    size = dirname.split('_')[2]
    # Handle special case for clone
    if size.endswith('clone'):
        size = size[:-6]
    return int(size)

# Get all directories and sort them by model size
base_dir = Path('ttt_output_complete')
model_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
model_dirs = sorted(model_dirs, key=lambda x: get_model_size(x.name))

# Store results for each model
all_results = {}
difficulty_levels = ['easy', 'medium', 'hard', 'expert']

for model_dir in model_dirs:
    model_size = get_model_size(model_dir.name)
    
    # Skip clone directory
    if model_dir.name.endswith('clone'):
        continue
        
    # Read task info
    df = pd.read_csv(model_dir / 'task_info.csv')
    
    # Calculate success rate per difficulty level
    success_by_level = df.groupby('level')['correct'].mean() * 100
    all_results[model_size] = success_by_level

# Convert results to DataFrame
results_df = pd.DataFrame(all_results).T
results_df.index.name = 'Training Examples'

# Create visualization
plt.figure(figsize=(12, 8))
for level in difficulty_levels:
    plt.plot(results_df.index, results_df[level], marker='o', label=level)

plt.xscale('log')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.xlabel('Number of Training Examples')
plt.ylabel('Success Rate (%)')
plt.title('Model Performance by Difficulty Level')
plt.legend()
plt.tight_layout()
plt.savefig('model_performance.png')

# Print statistics
print("\nModel Performance Statistics:")
print("\nSuccess rates by model size and difficulty level:")
print(results_df.round(1))

print("\nOverall average success rate by difficulty level:")
print(results_df.mean().round(1))

# Find best performing model for each difficulty level
best_models = results_df.idxmax()
print("\nBest performing model size for each difficulty level:")
for level in difficulty_levels:
    print(f"{level}: {best_models[level]} training examples "
          f"({results_df[level].max():.1f}% success rate)")



