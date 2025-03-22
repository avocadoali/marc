import pandas as pd
import os
import argparse

def combine_results(base_path, output_path):
    # Create a list to store dataframes for each epoch
    dfs = []

    # Read the first file to get task_id and level columns
    df_base = pd.read_csv(os.path.join(base_path, "adapters_json_ep_0_iter_-1/task_info.csv"))
    result_df = df_base[['task_id', 'level']]

    # add special epoch 9998 first    
    file_path = os.path.join(base_path, f"adapters_json_ep_9998_iter_-1/task_info.csv")
    print(file_path)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Add the 'correct' column with the epoch number
        result_df[f'correct_ep_0'] = df['correct']

    # Loop through each epoch (0 to 15)
    for epoch in range(16):
        file_path = os.path.join(base_path, f"adapters_json_ep_{epoch}_iter_-1/task_info.csv")
        print(file_path)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add the 'correct' column with the epoch number
            result_df[f'correct_ep_{epoch+1}'] = df['correct']

    # Save the combined dataframe
    print(f"Saving combined results to {output_path}")
    result_df.to_csv(output_path, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--base_path", 
                        type=str, 
                        default="experiments_submission_files/epoch_scaling_complete_rerun/epoch_scaling_barc/epoch_scaling_barc_non_finetuned_output",
                        help="Path to the base directory containing the epoch folders")
    parser.add_argument("--output_path", 
                        type=str, 
                        default="evaluation/data_es/",
                        help="Path to the output file")

    args = parser.parse_args()

    base_path = args.base_path

    output_path = args.output_path
    out_filename = base_path.split("/")[-1] + "_combined_results.csv"
    print(f'out_filename: {out_filename}')


    combine_results(base_path, os.path.join(output_path, out_filename))
