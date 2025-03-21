import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
sns.set_style("whitegrid")

# Function to create plot for each dataset
def create_plot(data_path, ax, title):
    # Read the CSV file
    df = pd.read_csv(data_path)
    
    # Calculate success rate for each difficulty level and epoch
    epoch_cols = [col for col in df.columns if col.startswith('correct_ep_')]
    
    success_rates = []
    for difficulty in ['easy', 'medium', 'hard', 'expert']:
        difficulty_mask = df['level'] == difficulty
        for col in epoch_cols:
            success_rate = df[difficulty_mask][col].mean() * 100
            success_rates.append({
                'Difficulty': difficulty,
                'Epoch': int(col.split('_')[-1]),
                'Success Rate (%)': success_rate
            })

    plot_df = pd.DataFrame(success_rates)

    # Create line plot with different markers for each difficulty
    for difficulty, marker in zip(['easy', 'medium', 'hard', 'expert'], ['o', 's', '^', 'D']):
        mask = plot_df['Difficulty'] == difficulty
        sns.lineplot(
            data=plot_df[mask],
            x='Epoch',
            y='Success Rate (%)',
            marker=marker,
            label=difficulty.capitalize(),
            markersize=8,
            ax=ax
        )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.legend(title='Difficulty', title_fontsize=12, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.7)

# Create plots for both datasets
create_plot('evaluation/data_es/epoch_scaling_barc_non_finetuned_output_combined_results.csv', 
           ax1, 'Non-Finetuned Model Performance')
create_plot('evaluation/data_es/epoch_scaling_barc_finetuned_output_combined_results.csv', 
           ax2, 'Finetuned Model Performance')

# Add a super title
fig.suptitle('Model Performance Comparison Across Training Epochs', fontsize=16, y=1.05)

# Adjust layout and save
plt.tight_layout()
plt.savefig('evaluation/plots_es/training_progression_comparison_es_marc_barc.png', 
            dpi=300, bbox_inches='tight')
plt.close()
