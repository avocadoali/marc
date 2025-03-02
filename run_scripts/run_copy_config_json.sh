#!/bin/bash

# Define the source file and the target directory
# SOURCE_FILE="experiments_thesis/2k_test_run/adapters_json/0a1d4ef5/config.json"
# SOURCE_FILE="experiments_thesis/70_test_run/adapters_json/0a1d4ef5/config.json"

# EXPERIMENT_FOLDER="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_1"
EXPERIMENT_FOLDER="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_2"
# EXPERIMENT_FOLDER="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_3"

SOURCE_FILE="${EXPERIMENT_FOLDER}/adapters_json/0a1d4ef5/config.json"
# SECOND_SOURCE_FILE="${EXPERIMENT_FOLDER}/adapters_json/00576224/config.json"


# iterate over all directories in 2k_test_run/ with a loop
# for dir in experiments_thesis/70_test_run/*; do

for dir in ${EXPERIMENT_FOLDER}/*; do
    # Skip the adapters_json directory
    if [ -d "$dir" ] && [ "$dir" != "${EXPERIMENT_FOLDER}/adapters_json" ]; then
        TARGET_DIR="$dir"

        # Iterate over each directory in the target directory
        for subdir in "$TARGET_DIR"/*; do
            if [ -d "$subdir" ]; then
                # Copy the source file into the current directory
                cp "$SOURCE_FILE" "$subdir/"
                # cp "$SECOND_SOURCE_FILE" "$subdir/"
            fi
        done

        echo "Copied config.json to $dir"

    fi
done


