#!/bin/bash

# Define the source file and the target directory
# SOURCE_FILE="experiments_thesis/2k_test_run/adapters_json/0a1d4ef5/config.json"
SOURCE_FILE="experiments_thesis/70_test_run/adapters_json/0a1d4ef5/config.json"

# iterate over all directories in 2k_test_run/ with a loop
for dir in experiments_thesis/70_test_run/*; do
    # Copy the source file into the current directory
    # skip the adapters_json directory
    if [ -d "$dir" ] && [ "$dir" != "experiments_thesis/70_test_run/adapters_json" ]; then

        TARGET_DIR="$dir"

        # Iterate over each directory in the target directory
        for dir in "$TARGET_DIR"/*; do
            if [ -d "$dir" ]; then
                # Copy the source file into the current directory
            cp "$SOURCE_FILE" "$dir/"
                echo "Copied config.json to $dir"
            fi
        done
    fi
done
