

# For Dataset Scaling

# base_path="experiments_submission_files/epoch_scaling_complete_rerun/epoch_scaling_barc/epoch_scaling_barc_non_finetuned_output"
# base_path="experiments_submission_files/epoch_scaling_complete_rerun/epoch_scaling_barc/epoch_scaling_barc_finetuned_output"

base_path="experiments_submission_files/experiments_thesis_ds_barc/ds_barc_finetuned_output"
# base_path="experiments_submission_files/experiments_thesis_ds_barc/ds_barc_non_finetuned_output"

output_path="evaluation/data_ds/"

python evaluation/scripts/combined_results.py --base_path ${base_path} --output_path ${output_path}
