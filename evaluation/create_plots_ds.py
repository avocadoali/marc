import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
# Read the CSV data


def create_plots(base_path, output_name):
    dict_metrics = {}

    for x in [20, 125, 300, 500]:
        df = pd.read_csv(f'{base_path}/adapters_json_ep_0_iter_{x}/task_info.csv')
        # Calculate metrics
        metrics = {
        'total_solved': len(df[df['correct'] == True]),
        'total_solved_easy': len(df[(df['level'] == 'easy') & (df['correct'] == True)]),
        'total_solved_medium': len(df[(df['level'] == 'medium') & (df['correct'] == True)]),
        'total_solved_hard': len(df[(df['level'] == 'hard') & (df['correct'] == True)]),
        'total_solved_expert': len(df[(df['level'] == 'expert') & (df['correct'] == True)]),

        'accuracy_easy': len(df[(df['level'] == 'easy') & (df['correct'] == True)]) / len(df[df['level'] == 'easy']),
        'accuracy_medium': len(df[(df['level'] == 'medium') & (df['correct'] == True)]) / len(df[df['level'] == 'medium']),
        'accuracy_hard': len(df[(df['level'] == 'hard') & (df['correct'] == True)]) / len(df[df['level'] == 'hard']),
        'accuracy_expert': len(df[(df['level'] == 'expert') & (df['correct'] == True)]) / len(df[df['level'] == 'expert'])
        }   
        dict_metrics[x] = metrics


    print(dict_metrics)

    # Prepare data for plotting
    iterations = list(dict_metrics.keys())

    # Create one figure with three subplots
    plt.figure(figsize=(20, 6))

    # Plot 1: Total accuracy vs iterations
    plt.subplot(131)
    total_accuracy = [metrics['total_solved'] / len(df) for x, metrics in dict_metrics.items()]
    plt.plot(iterations, total_accuracy, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Total Accuracy')
    plt.title('Total Accuracy vs Iterations')
    plt.grid(True)

    # Plot 2: Number of solved problems by difficulty
    plt.subplot(132)
    for difficulty in ['easy', 'medium', 'hard', 'expert']:
        solved = [metrics[f'total_solved_{difficulty}'] for metrics in dict_metrics.values()]
        plt.plot(iterations, solved, marker='o', label=difficulty.capitalize())
    plt.xlabel('Iterations')
    plt.ylabel('Number of Solved Problems')
    plt.title('Number of Solved Problems\nby Difficulty vs Iterations')
    plt.legend()
    plt.grid(True)

    # Plot 3: Accuracy by difficulty
    plt.subplot(133)
    for difficulty in ['easy', 'medium', 'hard', 'expert']:
        accuracy = [metrics[f'accuracy_{difficulty}'] for metrics in dict_metrics.values()]
        plt.plot(iterations, accuracy, marker='o', label=difficulty.capitalize())
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Difficulty\nvs Iterations')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the combined figure
    plt.savefig(f'evaluation/plots/{output_name}_metrics.png')
    plt.close()



def __main__(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="experiments_submission_files/experiments_thesis_ds_barc/ds_barc_finetuned_output")
    args = parser.parse_args()
    base_path = args.base_path  
    output_name = base_path.split('/')[-1]
    create_plots(base_path, output_name)

if __name__ == "__main__":
    __main__()