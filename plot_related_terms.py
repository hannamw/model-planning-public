#%%
import json
import matplotlib.pyplot as plt
#%%
# Load the JSON file
with open('a_an/results/relevant_nodes_refined/Qwen3-14B/a-banker.json', 'r') as f:
    data = json.load(f)

# Extract related_terms_count values from each node
related_terms_counts = []
for node_id, node_data in data['feature_counts'].items():
    related_terms_counts.append(node_data['related_terms_count'])

# Sort in descending order
related_terms_counts.sort(reverse=True)

# Create x-axis indices (0-based)
x_indices = list(range(len(related_terms_counts)))

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(x_indices, related_terms_counts, linewidth=2)

# Customize the plot
plt.title('Related Terms Count Distribution', fontsize=14)
plt.xlabel('Node Index (sorted by count)', fontsize=12)
plt.ylabel('Related Terms Count', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.show()
# %%
related_terms_counts[:50]
# %%
