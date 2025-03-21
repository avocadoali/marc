import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from the two runs
ekin_data = {
    'easy': 108,
    'medium': 65,
    'hard': 20,
    'expert': 6
}

barc_data = {
    'easy': 104,
    'medium': 71,
    'hard': 29,
    'expert': 8
}

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Set the positions of the bars
levels = ['easy', 'medium', 'hard', 'expert']
x = np.arange(len(levels))
width = 0.35

# Create bars
plt.bar(x - width/2, [ekin_data[level] for level in levels], width, 
        label=f'MARC (Overall: {199/400:.1%})', color='lightcoral')
plt.bar(x + width/2, [barc_data[level] for level in levels], width, 
        label=f'BARC (Overall: {212/400:.1%})', color='skyblue')

# Customize the plot
plt.xlabel('Difficulty Level')
plt.ylabel('Number of Solved Tasks')
plt.title('Baseline Performance Comparison between MARC and BARC')
plt.xticks(x, levels)
plt.legend()

# Add value labels on top of each bar
for i in x:
    plt.text(i - width/2, ekin_data[levels[i]], str(ekin_data[levels[i]]), 
             ha='center', va='bottom')
    plt.text(i + width/2, barc_data[levels[i]], str(barc_data[levels[i]]), 
             ha='center', va='bottom')

# Remove the figtext and add tight_layout
plt.tight_layout()
plt.savefig('evaluation/plots_baseline/baseline_comparison.png')
plt.close()
