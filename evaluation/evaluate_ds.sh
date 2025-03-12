#!/bin/bash



base_path="experiments_submission_files/experiments_thesis_ds_barc/ds_barc_non_finetuned_output"

# Create CSV header
echo "epoch,total_accuracy,easy,medium,hard,expert" >> ${base_path}/level_results.csv

evaluate_submission() {
    local x=$1
    local base_path=$2
    local submission_path="${base_path}/adapters_json_ep_0_iter_${x}/submission_default.json"

    echo "Evaluating epoch ${x}..."
    echo "Submission path: ${submission_path}"
    
    # Run evaluation and capture output
    output=$(python -m arclib.eval \
        --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
        --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
        --submission_file "${submission_path}")
    
    # Extract values using grep and awk
    accuracy=$(echo "$output" | grep "Competition Accuracy:" | awk -F'= ' '{print $2}')
    easy=$(echo "$output" | grep "easy" | awk '{print $2}')
    medium=$(echo "$output" | grep "medium" | awk '{print $2}')
    hard=$(echo "$output" | grep "hard" | awk '{print $2}')
    expert=$(echo "$output" | grep "expert" | awk '{print $2}')
    
    # Append to CSV
    echo "${x},${accuracy},${easy},${medium},${hard},${expert}" >> ${base_path}/level_results.csv
    
    # Print original output
    echo "$output"
    echo
}

for x in 20 125 300 500; do
    evaluate_submission $x $base_path
done
