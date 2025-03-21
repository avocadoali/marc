#!/bin/bash


# BARC baseline
python -m arclib.eval \
    --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
    --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
    --submission_file experiments_submission_files/baseline_barc_output/submission_default.json

# EKIN baseline
python -m arclib.eval \
    --data_file ./arc-prize-2024/arc-agi_evaluation_challenges.json \
    --solution_file ./arc-prize-2024/arc-agi_evaluation_solutions.json \
    --submission_file experiments_submission_files/baseline_ekin_output/submission_default.json


