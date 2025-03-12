# only run for ttt_adapters_250
# python -m arclib.eval \
#     --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
#     --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
#     --submission_file "/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_2_output/adapters_json_ep_1_iter_-1/submission_default.json"








# # Loop through all directories in ttt_output_complete that match the pattern ttt_output_*
# for dir in ttt_output_complete/ttt_output_*; do
#     # Skip if it's not a directory
#     [ ! -d "$dir" ] && continue
    
#     echo "Processing $dir..."
#     python -m arclib.eval \
#     --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
#     --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
#     --plot_mistakes \
#     --submission_file "$dir/submission.json"
# done

#!/bin/bash

evaluate_submission() {
    local epoch=$1
    local base_path="/hkfs/work/workspace/scratch/tum_ind3695-arc-workspace/experiments_thesis_epoch_scaling/baseline_epoch_scaling_batch_3_output"
    local submission_path="${base_path}/adapters_json_ep_${epoch}_iter_-1/submission_default.json"


    echo "Evaluating epoch ${epoch}..."
    echo "Submission path: ${submission_path}"
    python -m arclib.eval \
        --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
        --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
        --submission_file "${submission_path}"
}

# Loop through epochs 1 to 3
for epoch in {0..3}; do
    evaluate_submission $epoch
    # new line
    echo
done
