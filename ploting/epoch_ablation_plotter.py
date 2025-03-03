import matplotlib.pyplot as plt
import numpy as np

# Data from the tables
epochs = [1, 2, 3]

# Structure: batch_size -> difficulty -> values
data = {
    'Batch=1': {
        'easy': [105, 102, 102],
        'medium': [59, 62, 57],
        'hard': [21, 21, 22],
        'expert': [6, 6, 6]
    },
    'Batch=2': {
        'easy': [104, 105, 108],
        'medium': [62, 61, 61],
        'hard': [20, 23, 23],
        'expert': [7, 6, 6]
    },
    'Batch=3': {
        'easy': [96, 102, 102],
        'medium': [62, 63, 66],
        'hard': [22, 21, 22],
        'expert': [6, 6, 6]
    }
}

# Create the plot with reduced width
plt.figure(figsize=(10, 8))

# Colors and line styles
colors = {
    'easy': 'blue',
    'medium': 'orange',
    'hard': 'green',
    'expert': 'red'
}

line_styles = {
    'Batch=1': '-',
    'Batch=2': '--',
    'Batch=3': ':'
}

# Plot each difficulty level for each batch
for batch_name, batch_data in data.items():
    for difficulty, values in batch_data.items():
        # Convert to percentage
        percentages = [v * 100 / 400 for v in values]
        # Don't include difficulty in label, just batch
        plt.plot(epochs, percentages, 
                marker='o',
                linestyle=line_styles[batch_name],
                color=colors[difficulty],
                label=batch_name)  # Only include batch in label

# Create custom legend elements
from matplotlib.lines import Line2D

# First legend - Difficulty levels (colors)
difficulty_legend_elements = [
    Line2D([0], [0], color=color, label=diff.capitalize(), marker='o')
    for diff, color in colors.items()
]

# Second legend - Batch sizes (line styles)
batch_legend_elements = [
    Line2D([0], [0], color='black', label=batch, linestyle=style, marker='o')
    for batch, style in line_styles.items()
]

# Combine both sets of legend elements
all_legend_elements = difficulty_legend_elements + batch_legend_elements

# Add single legend with all elements
plt.legend(handles=all_legend_elements, 
          loc='upper left', 
          bbox_to_anchor=(1.05, 1.0),
          title='Difficulty & Batch Size')

# Customize the plot
plt.title('Model Performance by Difficulty Level and Batch Size')
plt.xlabel('Epoch')
plt.ylabel('Success Rate (%)')
plt.grid(True, linestyle='--', alpha=0.7)

# Set y-axis limits
plt.ylim(0, 30)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.82)

# Save the figure
plt.savefig('experiment_plots/epoch_ablation_plot.png', bbox_inches='tight', dpi=300)