# only run for ttt_adapters_250
python -m arclib.eval \
    --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
    --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
    --submission_file "ttt_output_llama3/ttt_output_llama3_70_epochs_1/submission.json"
    # --submission_file "ttt_output_complete/ttt_output_50_epochs_1/submission.json"
    # --submission_file "ttt_output_complete/ttt_output_250/submission.json"
    # --plot_mistakes \
    # --submission_file "ttt_output_complete/ttt_output_50_epochs_1/submission.json"


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
