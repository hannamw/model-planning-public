#%%
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# Read the JSON file
with open('results/couplet_samples_manual_lowtemp/Qwen3-14B.json', 'r') as f:
    data = json.load(f)

# For each position (1st most common, 2nd most common, etc), store all percentages
position_percentages = defaultdict(list)

# Process each example's distribution
for example in data:
    dist = example['distribution']
    total_samples = example['num_samples']
    
    # Convert to percentages and sort by frequency
    percentages = [(count/total_samples * 100) for count in dist.values()]
    percentages.sort(reverse=True)
    
    # Store each position's percentage
    for i, pct in enumerate(percentages):
        position_percentages[i].append(pct)

# Calculate statistics for each position
print("\nDistribution Statistics:")
print("Position | Mean % | Median % | Std Dev")
print("-" * 40)

max_positions = 10  # Show top 10 positions
means = []
medians = []
stds = []
positions = []

for pos in range(max_positions):
    if pos in position_percentages:
        values = position_percentages[pos]
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values)
        means.append(mean)
        medians.append(median)
        stds.append(std)
        positions.append(pos + 1)
        print(f"{pos+1:8d} | {mean:6.2f} | {median:8.2f} | {std:6.2f}")

# Calculate cumulative statistics
print("\nCumulative Coverage Statistics:")
print("Top-K | Mean Cumulative % | Median Cumulative %")
print("-" * 45)

cumulative_means = []
cumulative_medians = []

for k in range(1, max_positions + 1):
    cumulative_percentages = []
    for example in data:
        dist = example['distribution']
        total_samples = example['num_samples']
        
        # Get top-k percentages
        percentages = sorted([(count/total_samples * 100) for count in dist.values()], reverse=True)
        cumulative = sum(percentages[:k])
        cumulative_percentages.append(cumulative)
    
    mean_cum = np.mean(cumulative_percentages)
    median_cum = np.median(cumulative_percentages)
    cumulative_means.append(mean_cum)
    cumulative_medians.append(median_cum)
    print(f"{k:5d} | {mean_cum:16.2f} | {median_cum:18.2f}")

# Create visualization
plt.style.use('seaborn-v0_8')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Individual position percentages
ax1.bar(positions, means, yerr=stds, capsize=5, alpha=0.7)
ax1.set_xlabel('Position in Distribution')
ax1.set_ylabel('Average Percentage')
ax1.set_title('Average Percentage per Position')
ax1.grid(True, alpha=0.3)

# Plot 2: Cumulative distribution
ax2.plot(positions, cumulative_means, 'b-', marker='o', label='Mean')
ax2.plot(positions, cumulative_medians, 'r--', marker='s', label='Median')
ax2.set_xlabel('Top-K Positions')
ax2.set_ylabel('Cumulative Percentage')
ax2.set_title('Cumulative Coverage')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('distribution_analysis.png')
plt.close() 
# %%
