import json

# Read the submission file
with open('ttt_output/submission.json', 'r') as f:
    submission_data = json.load(f)

# Read the all_predictions file
with open('ttt_output/all_predictions.json', 'r') as f:
    predictions_data = json.load(f)

# Get stats for submission.json
submission_count = len(submission_data)
attempts_per_entry = [len(entry) for entry in submission_data.values()]
total_attempts = sum(attempts_per_entry)

# Get stats for all_predictions.json
predictions_count = len(predictions_data)

print("=== Submission.json Stats ===")
print(f"Number of unique IDs: {submission_count}")
print(f"Total number of attempts: {total_attempts}")
print(f"Average attempts per ID: {total_attempts/submission_count:.2f}")

print("\n=== All_predictions.json Stats ===")
print(f"Number of entries: {predictions_count}")
