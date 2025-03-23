import pandas as pd
import argparse

def create_oracle_version(input_path, output_path):
    # Read the original CSV
    df = pd.read_csv(input_path)

    # Get all columns that start with 'correct_iter_'
    iter_cols = [col for col in df.columns if col.startswith('correct_iter_')]

    # Sort the iteration columns to ensure we process them in order
    iter_cols.sort(key=lambda x: int(x.split('_')[-1]))

    # For each epoch column (except the first one)
    for i in range(1, len(iter_cols)):
        # If any previous iteration was True, make current iteration True
        df[iter_cols[i]] = df[[iter_cols[i]] + iter_cols[:i]].any(axis=1)

    # Save the oracle version
    df.to_csv(output_path, index=False)



def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", 
                        type=str, 
                        default='evaluation/data_es/epoch_scaling_barc_finetuned_output_combined_results.csv')
    parser.add_argument("--output_path", 
                        type=str, 
                        default='evaluation/data_es/epoch_scaling_barc_finetuned_output_combined_results_oracle.csv')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path


    # For Dataset Scaling
    input_path = "evaluation/data_ds/ds_barc_finetuned_output_combined_results.csv"
    output_path = "evaluation/data_ds/ds_barc_finetuned_output_combined_results_oracle.csv"

    # input_path = "evaluation/data_ds/ds_barc_non_finetuned_output_combined_results.csv"
    # output_path = "evaluation/data_ds/ds_barc_non_finetuned_output_combined_results_oracle.csv"


    create_oracle_version(input_path, output_path)


if __name__ == "__main__":
    __main__()