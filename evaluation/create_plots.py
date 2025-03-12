import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def main(df, name):
    # Set seaborn style
    sns.set_theme()  # This is the preferred way to set seaborn styling
    sns.set_palette("husl")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Total accuracy over epochs
    ax1.plot(df['epoch'], df['total_accuracy'], marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Accuracy')
    ax1.set_title('Total Accuracy vs Epoch')
    ax1.grid(True)

    # Plot 2: Difficulty levels over epochs
    difficulties = ['easy', 'medium', 'hard', 'expert']
    for diff in difficulties:
        ax2.plot(df['epoch'], df[diff], marker='o', label=diff.capitalize(), linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Number of Solved Problems')
    ax2.set_title('Problems Solved by Difficulty')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'evaluation/plots/{name}_level_results.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="experiments_submission_files/experiments_thesis_ds_barc")
    args = parser.parse_args()

    # Read the CSV file
    # experiment_name = "ds_barc_non_finetuned"

    name = args.base_path.split("/")[-1]
    df = pd.read_csv(f"{args.base_path}/level_results.csv")
    main(df, name)
