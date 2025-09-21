#%%
#!/usr/bin/env python3
"""
Plot probing test F1 scores across layers for all models.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_probing_results(model_name):
    """Load probing results from JSON file."""
    json_path = Path(f"{model_name}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    layers = []
    f1_scores = []
    
    for result in data:
        layers.append(result['layer'])
        # Use macro F1 score as it's more balanced across classes
        f1_scores.append(result['f1_scores']['macro'])
    
    return layers, f1_scores

#%%
# Models in the specified order: Qwen3 by size, then Llama
models = [
    ("Qwen3-0.6B", "Qwen3 0.6B"),
    ("Qwen3-1.7B", "Qwen3 1.7B"), 
    ("Qwen3-4B", "Qwen3 4B"),
    ("Qwen3-8B", "Qwen3 8B"),
    ("Qwen3-14B", "Qwen3 14B"),
    ("Qwen3-32B", "Qwen3 32B"),
    ("Meta-Llama-3-8B-Instruct", "Llama-3 8B Instruct")
]

plt.figure(figsize=(12, 6))

# Colors for different models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf', '#8c564b']

for i, (model_file, model_label) in enumerate(models):
    try:
        layers, f1_scores = load_probing_results(model_file)
        plt.plot(layers, f1_scores, marker='o', linewidth=2, markersize=4, 
                color=colors[i], label=model_label)
        print(f"Loaded {model_label}: {len(layers)} layers")
    except FileNotFoundError:
        print(f"Warning: Could not find results for {model_file}")
        continue

plt.xlabel('Layer', fontsize=16)
plt.ylabel('Test F1 Score (Macro)', fontsize=16)
plt.title('Probing Test F1 Scores Across Layers', fontsize=18, fontweight='bold')
plt.legend(loc='lower right', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_path = Path("f1_scores_by_layer.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

plt.show()
# %%
