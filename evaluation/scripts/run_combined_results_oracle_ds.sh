
# input_path="evaluation/data_es/epoch_scaling_barc_finetuned_output_combined_results.csv"
input_path="evaluation/data_es/epoch_scaling_barc_non_finetuned_output_combined_results.csv"

output_path="${input_path%.csv}_oracle.csv"


python evaluation/scripts/combined_results_oracle.py --input_path ${input_path} --output_path ${output_path}