import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up a single figure
fig, ax = plt.subplots(figsize=(12, 8))
sns.set_style("whitegrid")

# Function to create plot data
def create_plot_data(data_path):
    # Read the CSV file
    df = pd.read_csv(data_path)
    
    # Calculate success rate for each iteration
    epoch_cols = [col for col in df.columns if col.startswith('correct_iter_')]
    
    success_rates = []
    for col in epoch_cols:
        total_tasks = len(df)
        solved_count = df[col].sum()
        success_rate = (solved_count / total_tasks) * 100  # Convert to percentage
        success_rates.append({
            'Iteration': int(col.split('_')[-1]),
            'Success Rate (%)': success_rate
        })

    return pd.DataFrame(success_rates)

# Create plots for all datasets
non_finetuned_data = create_plot_data('evaluation/data_ds/ds_barc_non_finetuned_output_combined_results.csv')
finetuned_data = create_plot_data('evaluation/data_ds/ds_barc_finetuned_output_combined_results.csv')
finetuned_oracle_data = create_plot_data('evaluation/data_ds/ds_barc_finetuned_output_combined_results_oracle.csv')
non_finetuned_oracle_data = create_plot_data('evaluation/data_ds/ds_barc_non_finetuned_output_combined_results_oracle.csv')

# Plot all lines on the same axes
sns.lineplot(
    data=finetuned_oracle_data,
    x='Iteration',
    y='Success Rate (%)',
    marker='o',
    markersize=8,
    label='Finetuned Oracle Model',
    color='green',  # Green with transparency for finetuned oracle
    alpha=0.5,
    ax=ax
)

sns.lineplot(
    data=finetuned_data,
    x='Iteration',
    y='Success Rate (%)',
    marker='o',
    markersize=8,
    label='Finetuned Model',
    color='darkgreen',  # Dark green for finetuned
    ax=ax
)

sns.lineplot(
    data=non_finetuned_oracle_data,
    x='Iteration',
    y='Success Rate (%)',
    marker='o',
    markersize=8,
    label='Non-Finetuned Oracle Model',
    color='blue',  # Blue with transparency for non-finetuned oracle
    alpha=0.5,
    ax=ax
)

sns.lineplot(
    data=non_finetuned_data,
    x='Iteration',
    y='Success Rate (%)',
    marker='o',
    markersize=8,
    label='Non-Finetuned Model',
    color='darkblue',  # Dark blue for non-finetuned
    ax=ax
)




# Set title and labels
ax.set_title('BARC: Dataset Scaling Performance', fontsize=14, pad=20)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=10, loc='lower right')

# Adjust layout and save
plt.tight_layout()
plt.savefig('evaluation/plots_ds/barc_ds_performance.png', 
            dpi=300, bbox_inches='tight')
plt.close()



# Save all datapoints to CSV
combined_data = pd.DataFrame({
    'Iteration': non_finetuned_data['Iteration'],
    'Non-Finetuned': non_finetuned_data['Success Rate (%)'],
    'Non-Finetuned Oracle': non_finetuned_oracle_data['Success Rate (%)'],
    'Finetuned': finetuned_data['Success Rate (%)'],
    'Finetuned Oracle': finetuned_oracle_data['Success Rate (%)']
})
combined_data.to_csv('evaluation/plots_ds/barc_ds_performance_data.csv', index=False)


