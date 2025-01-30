import json
import matplotlib.pyplot as plt

# Read the JSON data
with open('stats/predict_stats.json', 'r') as f:
    data = json.load(f)

# Extract training sizes and accuracy
sizes = []
accuracies = []

for entry in data:
    if entry["Max Training Size"] and "ttt_adapters_" in entry["Max Training Size"]:
        # Extract the number from strings like "ttt_adapters_50"
        size = int(entry["Max Training Size"].split('_')[-1])
        accuracy = (entry["correct prediction"] / entry["total prediction"]) * 100
        
        sizes.append(size)
        accuracies.append(accuracy)

# Sort the data points by size
sorted_points = sorted(zip(sizes, accuracies))
sizes, accuracies = zip(*sorted_points)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, accuracies, 'bo-')
plt.xscale('log')  # Use log scale for x-axis since sizes vary widely
plt.grid(True)

# Add labels and title
plt.xlabel('Training Size')
plt.ylabel('Accuracy (%)')
plt.title('Prediction Accuracy vs Training Size')

# Add value labels on the points
for i, (size, acc) in enumerate(zip(sizes, accuracies)):
    plt.annotate(f'{acc:.1f}%', 
                (size, acc),
                textcoords="offset points",
                xytext=(0,10),
                ha='center')

plt.tight_layout()
# save the plot

plt.savefig('plots/predict_accuracy.png')