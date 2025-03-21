# ## Experiments
# # Barc
# EXPERIMENT_NAME="experiments_thesis_epoch_scaling_8_barc"
# # EXPERIMENT_NAME="experiments_thesis_ds_barc"

# # Ekin
# # EXPERIMENT_NAME="experiments_thesis_epoch_scaling_8"

# Dataset scaling 20k
# EXPERIMENT_NAME="experiments_thesis_dataset_scaling"

# DST_DIR="experiments_submission_files/${EXPERIMENT_NAME}"
# mkdir -p ${DST_DIR}
# # barc epoch scaling non finetuned model
# EXPERIMENT_FOLDER="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/${EXPERIMENT_NAME}"


# Rerun experiments

# epoch_scaling_complete_rerun/epoch_scaling_barc/epoch_scaling_barc_finetuned
EXPERIMENT_NAME="epoch_scaling_complete_rerun/epoch_scaling_barc/"
DST_DIR="experiments_submission_files/${EXPERIMENT_NAME}"
mkdir -p ${DST_DIR}
# barc epoch scaling non finetuned model
EXPERIMENT_FOLDER="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/${EXPERIMENT_NAME}"




# Iterate only through directories that end with _output
for dir in ${EXPERIMENT_FOLDER}/*_output*; do
    echo "Processing $dir"
    # and ends with _output
    if [ -d "$dir" ] && [[ "$dir" == *"_output" ]]; then
    # if [ -d "$dir" ] && [[ "$dir" == *"_output_subset" ]]; then
        TARGET_DIR="$dir"

        echo "Copying ${TARGET_DIR} to ${DST_DIR}/"
        # Use rsync to exclude logs directory
        rsync -av --exclude='logs/' ${TARGET_DIR} ${DST_DIR}/
        # Alternative using cp with find:
        # find ${TARGET_DIR} -mindepth 1 -not -path '*/logs/*' -exec cp -r {} ${DST_DIR}/ \;
        
        echo "Completed processing $dir"
    fi
done
