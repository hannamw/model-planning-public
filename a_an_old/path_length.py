#%%
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from circuit_tracer.graph import Graph, normalize_matrix
models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']

#%%
def load_model_results(model_name: str, results_dir: str = 'results/logit-lens'):
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    metadata_path = model_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata


def compute_path_length_influence(graph: Graph):
    n_features = len(graph.selected_features)
    n_pos = graph.n_pos
    n_logits = len(graph.logit_tokens)
    n_errors = n_pos * graph.cfg.n_layers
    adj = graph.adjacency_matrix
    adj[:, n_features: n_features + n_errors] = 0

    A = normalize_matrix(adj)

    influence_by_path_length = torch.zeros(graph.cfg.n_layers + 1)
    logit_weights = torch.zeros(A.shape[0], device=A.device)
    logit_weights[-n_logits:] = graph.logit_probabilities

    embed_weights = torch.zeros(A.shape[0], device=A.device)
    embed_weights[n_features + n_errors: n_features + n_errors + n_pos] = 1

    current_paths = embed_weights
    for i in range(graph.cfg.n_layers + 1):
        current_paths = A @ current_paths
        if not current_paths.any():
            break
        influence_by_path_length[i] = current_paths.dot(logit_weights)
        
    cumsum_path_influence = torch.cumsum(influence_by_path_length, -1)
    total = cumsum_path_influence[-1].item()
    if total != 0:
        cumsum_path_influence /= total
    return influence_by_path_length, cumsum_path_influence

#%%
# Store results for all models
all_model_results = {}

for model in models:
    print(f"Processing model: {model}")
    
    # Load model metadata
    # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    logit_lens_model_name = f"Qwen3-{size_part}"
    
    metadata = load_model_results(logit_lens_model_name)
    
    graph_dir = Path('attribution_graphs') / model
    
    # Initialize lists for different article types
    cumsum_path_influences = []
    path_influences = []
    
    # Process each example based on metadata
    for _, row in tqdm(metadata.iterrows()):
        correct_article = row['correct_articles']
        profession = row['professions']
        
        # Generate filename based on metadata
        filename = f"{correct_article}-{profession}.pt"
        graph_file = graph_dir / filename
        
        graph = Graph.from_pt(str(graph_file))
        influence_by_path_length, cumsum_path_influence = compute_path_length_influence(graph)
        
        # Add to appropriate lists
        cumsum_path_influences.append(cumsum_path_influence)
        path_influences.append(influence_by_path_length)
         
    cumsum_path_influences = torch.stack(cumsum_path_influences)
    path_influences = torch.stack(path_influences)
    
    # Create boolean masks for filtering
    is_a = metadata['correct_articles'] == 'a'
    is_an = metadata['correct_articles'] == 'an'
    # The 'correct?' column is assumed to exist and be boolean
    is_correct = metadata['correct?'] == True
    is_incorrect = ~is_correct

    def get_mean_influence(mask):
        subset = cumsum_path_influences[mask.values]
        # Return mean if examples exist, otherwise return zeros to avoid errors
        return subset.mean(0) if subset.shape[0] > 0 else torch.zeros_like(cumsum_path_influences[0])

    # Store results for this model, calculating means for each group
    all_model_results[model] = {
        'all': cumsum_path_influences.mean(0),
        'a': get_mean_influence(is_a),
        'an': get_mean_influence(is_an),
        'correct': get_mean_influence(is_correct),
        'incorrect': get_mean_influence(is_incorrect),
        'a_correct': get_mean_influence(is_a & is_correct),
        'a_incorrect': get_mean_influence(is_a & is_incorrect),
        'an_correct': get_mean_influence(is_an & is_correct),
        'an_incorrect': get_mean_influence(is_an & is_incorrect),
        'metadata': metadata,
        'per_example_cumsum_influences': cumsum_path_influences,
        'per_example_path_influences': path_influences
    }
    
    # Save results for this model
    results_dir = Path('results/path_length')
    results_dir.mkdir(exist_ok=True)
    torch.save(all_model_results[model], results_dir / f'{logit_lens_model_name}_path_length_results.pt')
    print(f"  Saved results to {results_dir / f'{logit_lens_model_name}_path_length_results.pt'}")
#%%
# Load results from disk for plotting
def load_saved_results(models):
    """Load saved results from disk"""
    loaded_results = {}
    results_dir = Path('results/path_length')
    
    for model in models:
        # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
        model_name_parts = model.split('-')
        size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
        if size_part.endswith('B'):
            size_part = size_part[:-1] + 'B'
        logit_lens_model_name = f"Qwen3-{size_part}"
        
        results_file = results_dir / f'{logit_lens_model_name}_path_length_results.pt'
        loaded_results[model] = torch.load(results_file, weights_only=False)
        print(f"Loaded results for {model} from {results_file}")
    
    return loaded_results

# Load results from disk
all_model_results = load_saved_results(models)

# %%
# Plot results for all models - by article type
fig, axes = plt.subplots(1, len(models), figsize=(20, 6), sharey=True)
if len(models) == 1:
    axes = [axes]

for i, model in enumerate(models):
    results = all_model_results[model]
    
    x_range = range(1, len(results['all']) + 1)
    
    axes[i].plot(x_range, results['all'].cpu().numpy(), 'b-', label='All examples', linewidth=2)
    axes[i].plot(x_range, results['a'].cpu().numpy(), 'r--', label="'a' examples", linewidth=2)
    axes[i].plot(x_range, results['an'].cpu().numpy(), 'g--', label="'an' examples", linewidth=2)
    
    axes[i].set_xlabel('Path Length')
    if i == 0:
        axes[i].set_ylabel('Cumulative Path Influence')
    axes[i].set_title(f'{model}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Path Length Influence by Article Type', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
# Plot results for all models - by correctness
fig, axes = plt.subplots(1, len(models), figsize=(20, 6), sharey=True)
if len(models) == 1:
    axes = [axes]

for i, model in enumerate(models):
    results = all_model_results[model]
    
    x_range = range(1, len(results['all']) + 1)
    
    axes[i].plot(x_range, results['all'].cpu().numpy(), 'k-', label='All examples', linewidth=2, alpha=0.7)
    axes[i].plot(x_range, results['correct'].cpu().numpy(), 'c-', label="Correct", linewidth=2)
    axes[i].plot(x_range, results['incorrect'].cpu().numpy(), 'm-', label="Incorrect", linewidth=2)
    
    axes[i].set_xlabel('Path Length')
    if i == 0:
        axes[i].set_ylabel('Cumulative Path Influence')
    axes[i].set_title(f'{model}')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

fig.suptitle('Path Length Influence by Correctness', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
# Plot results for all models - detailed breakdown
fig, axes = plt.subplots(2, len(models), figsize=(20, 10), sharey=True, sharex=True)
if len(models) == 1:
    axes = axes.reshape(2, 1)

for i, model in enumerate(models):
    results = all_model_results[model]
    x_range = range(1, len(results['all']) + 1)
    
    # 'a' examples plot
    ax_a = axes[0, i]
    ax_a.plot(x_range, results['a'].cpu().numpy(), 'r-', label="'a' All", linewidth=2, alpha=0.7)
    ax_a.plot(x_range, results['a_correct'].cpu().numpy(), 'r--', label="'a' Correct", linewidth=2)
    ax_a.plot(x_range, results['a_incorrect'].cpu().numpy(), 'r:', label="'a' Incorrect", linewidth=2)
    ax_a.set_title(f'{model} - "a" examples')
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    # 'an' examples plot
    ax_an = axes[1, i]
    ax_an.plot(x_range, results['an'].cpu().numpy(), 'g-', label="'an' All", linewidth=2, alpha=0.7)
    ax_an.plot(x_range, results['an_correct'].cpu().numpy(), 'g--', label="'an' Correct", linewidth=2)
    ax_an.plot(x_range, results['an_incorrect'].cpu().numpy(), 'g:', label="'an' Incorrect", linewidth=2)
    ax_an.set_xlabel('Path Length')
    ax_an.legend()
    ax_an.grid(True, alpha=0.3)

    if i == 0:
        ax_a.set_ylabel('Cumulative Path Influence')
        ax_an.set_ylabel('Cumulative Path Influence')

fig.suptitle('Detailed Path Length Influence Breakdown', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %%
# Plot all models on same plot - by article type (zoomed to 0-10)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Define colors for each model
colors = ['blue', 'red', 'green', 'orange', 'purple']
# Define line styles for each subcategory
line_styles = {
    'all': '-',      # solid
    'a': '--',       # dashed  
    'an': ':'        # dotted
}

# Create clean model names
def get_clean_model_name(model):
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    return f"Qwen3-{size_part}"

for i, model in enumerate(models):
    results = all_model_results[model]
    color = colors[i % len(colors)]
    clean_name = get_clean_model_name(model)
    
    # Limit x_range to 1-10
    x_range = range(1, min(11, len(results['all']) + 1))
    
    # Plot each subcategory with different line styles
    ax.plot(x_range, results['all'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['all'], 
            label=f'{clean_name} - All', linewidth=2)
    ax.plot(x_range, results['a'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['a'], 
            label=f'{clean_name} - "a"', linewidth=2)
    ax.plot(x_range, results['an'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['an'], 
            label=f'{clean_name} - "an"', linewidth=2)

ax.set_xlabel('Path Length')
ax.set_ylabel('Cumulative Path Influence')
ax.set_title('Path Length Influence by Article Type (All Models)')
ax.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=5)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.show()
#%%
# Plot all models on same plot - by correctness (zoomed to 0-10)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Define colors for each model
colors = ['blue', 'red', 'green', 'orange', 'purple']
# Define line styles for each subcategory
line_styles = {
    'all': '-',        # solid
    'correct': '--',   # dashed  
    'incorrect': ':'   # dotted
}

for i, model in enumerate(models):
    results = all_model_results[model]
    color = colors[i % len(colors)]
    clean_name = get_clean_model_name(model)
    
    # Limit x_range to 1-10
    x_range = range(1, min(11, len(results['all']) + 1))
    
    # Plot each subcategory with different line styles
    ax.plot(x_range, results['all'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['all'], 
            label=f'{clean_name} - All', linewidth=2, alpha=0.7)
    ax.plot(x_range, results['correct'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['correct'], 
            label=f'{clean_name} - Correct', linewidth=2)
    ax.plot(x_range, results['incorrect'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['incorrect'], 
            label=f'{clean_name} - Incorrect', linewidth=2)

ax.set_xlabel('Path Length')
ax.set_ylabel('Cumulative Path Influence')
ax.set_title('Path Length Influence by Correctness (All Models)')
ax.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=5)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.show()

# %%
# Plot all models on same plot - detailed breakdown (zoomed to 0-10)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Define colors for each model
colors = ['blue', 'red', 'green', 'orange', 'purple']
# Define line styles for each subcategory
line_styles = {
    'a_correct': '-',    # solid
    'a_incorrect': '--', # dashed
    'an_correct': ':',   # dotted
    'an_incorrect': '-.' # dash-dot
}

for i, model in enumerate(models):
    results = all_model_results[model]
    color = colors[i % len(colors)]
    clean_name = get_clean_model_name(model)
    
    # Limit x_range to 1-10
    x_range = range(1, min(11, len(results['all']) + 1))
    
    # Plot each subcategory with different line styles
    ax.plot(x_range, results['a_correct'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['a_correct'], 
            label=f'{clean_name} - "a" Correct', linewidth=2)
    ax.plot(x_range, results['a_incorrect'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['a_incorrect'], 
            label=f'{clean_name} - "a" Incorrect', linewidth=2)
    ax.plot(x_range, results['an_correct'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['an_correct'], 
            label=f'{clean_name} - "an" Correct', linewidth=2)
    ax.plot(x_range, results['an_incorrect'].cpu().numpy()[:len(x_range)], 
            color=color, linestyle=line_styles['an_incorrect'], 
            label=f'{clean_name} - "an" Incorrect', linewidth=2)

ax.set_xlabel('Path Length')
ax.set_ylabel('Cumulative Path Influence')
ax.set_title('Detailed Path Length Influence Breakdown (All Models)')
ax.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center', ncol=4)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.show()
# %%
