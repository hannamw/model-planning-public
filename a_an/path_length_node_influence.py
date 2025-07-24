#%%
# Import additional required modules
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from circuit_tracer.graph import Graph, normalize_matrix

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 10
LAST_ONLY = False
PATH_LENGTH_RESULTS_DIR = Path('results/path_length_last') if LAST_ONLY else Path('results/path_length')  # Directory for path length results

models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']
model_sizes = [28,28, 36, 36, 40]
#%%
# Mapping from model config names to logit lens names
model_to_logit_lens = {
    'qwen3-0.6b-relu-lowl0': 'Qwen3-0.6B',
    'qwen3-1.7b-relu-lowl0': 'Qwen3-1.7B',
    'qwen3-4b-relu': 'Qwen3-4B',
    'qwen3-8b-relu': 'Qwen3-8B',
    'qwen3-14b-relu-lowl0': 'Qwen3-14B'
}

def load_important_nodes(model_name: str, example_key: str, 
                        required_profession_count: int = 5,
                        required_related_terms_count: int = 10) -> List[List[int]]:
    """Load and filter important nodes based on profession and related terms counts
    
    Args:
        model_name: Name of the model (e.g. 'qwen3-0.6b-relu-lowl0')
        example_key: Example identifier (e.g. 'a-archaeologist')
        required_profession_count: Minimum profession count threshold
        required_related_terms_count: Minimum related terms count threshold
    
    Returns:
        List of [layer, feature, pos] lists for important nodes, or None if no nodes found
    """
    # Convert model name format
    qwen_name = model_to_logit_lens[model_name]
    
    # Load relevant nodes
    relevant_nodes_path = f'results/relevant_nodes_refined/{qwen_name}/{example_key}.json'
    with open(relevant_nodes_path) as f:
        relevant_nodes = json.load(f)


    # Filter nodes by profession and related terms counts
    filtered_nodes = []
    for node_id, data in relevant_nodes['feature_counts'].items():
        if (data['profession_count'] > required_profession_count or 
            data['related_terms_count'] > required_related_terms_count):
            # Convert node_id format from layer_pos_feature to [layer, feature, pos]
            layer, pos, feature = map(int, node_id.split('_'))
            filtered_nodes.append([layer, pos, feature])
    
    return filtered_nodes if filtered_nodes else None

#%%
def load_model_results(model_name: str, results_dir: str = 'results/logit-lens'):
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    metadata_path = model_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata


def compute_path_length_influence(graph: Graph, selected_nodes=None):
    n_features = len(graph.selected_features)
    n_pos = graph.n_pos
    n_logits = len(graph.logit_tokens)
    n_errors = n_pos * graph.cfg.n_layers
    adj = graph.adjacency_matrix
    adj[:, n_features: n_features + n_errors] = 0

    A = normalize_matrix(adj)

    # Create selected nodes mask if provided
    selected_mask = torch.zeros_like(A[0])
    if not selected_nodes:
        selected_nodes = None
    else:
        candidate_features = graph.active_features[graph.selected_features].unsqueeze(1)
        selected_features = torch.tensor(selected_nodes).unsqueeze(0)
        matches = torch.all(candidate_features == selected_features, dim=2)
        selected_nodes_mask = torch.any(matches, dim=1)  # this is of size graph.selected_features < A.size(0)
        if LAST_ONLY:
            selected_nodes_mask &= graph.active_features[graph.selected_features][:, 1] == (graph.n_pos - 1)
        selected_mask[:len(selected_nodes_mask)] = selected_nodes_mask

        if not selected_mask.any():
            selected_nodes = None

    
    non_selected_influence_by_path_length = torch.zeros(graph.cfg.n_layers + 1)
    selected_influence_by_path_length = torch.zeros(graph.cfg.n_layers + 1)
    logit_weights = torch.zeros(A.shape[0], device=A.device)
    logit_weights[-n_logits:] = graph.logit_probabilities

    embed_weights = torch.zeros(A.shape[0], device=A.device)
    embed_weights[n_features + n_errors: n_features + n_errors + n_pos] = 1

    current_paths = embed_weights
    selected_paths = torch.zeros_like(embed_weights)
    for i in range(graph.cfg.n_layers + 1):
        current_paths = A @ current_paths
        selected_paths = A @ selected_paths
        if not current_paths.any():
            break
        non_selected_influence_by_path_length[i] = current_paths.dot(logit_weights)
        selected_influence_by_path_length[i] = selected_paths.dot(logit_weights)
        if selected_nodes is not None:
            selected_paths += current_paths * selected_mask
            current_paths *= (1 - selected_mask)
        
    cumsum_non_selected_path_influence = torch.cumsum(non_selected_influence_by_path_length, -1)
    cumsum_selected_path_influence = torch.cumsum(selected_influence_by_path_length, -1)

    total_path_influence = non_selected_influence_by_path_length + selected_influence_by_path_length
    cumsum_total_path_influence = cumsum_non_selected_path_influence + cumsum_selected_path_influence
    
    total = cumsum_total_path_influence[-1].item()
    if total != 0:
        cumsum_total_path_influence /= total
        cumsum_non_selected_path_influence /= total
        cumsum_selected_path_influence /= total
    return total_path_influence, cumsum_total_path_influence, non_selected_influence_by_path_length, cumsum_non_selected_path_influence, selected_influence_by_path_length, cumsum_selected_path_influence 

#%%
# Store results for all models
all_model_results = {}

# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
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
    non_selected_influences = []
    cumsum_non_selected_influences = []
    selected_influences = []
    cumsum_selected_influences = []
    
    # Process each example based on metadata
    for _, row in tqdm(metadata.iterrows()):
        correct_article = row['correct_articles']
        profession = row['professions']
        
        # Generate filename based on metadata
        filename = f"{correct_article}-{profession}.pt"
        graph_file = graph_dir / filename
        
        # Get important nodes for this example
        example_key = f"{correct_article}-{profession}"
        selected_nodes = load_important_nodes(model, example_key)
        
        graph = Graph.from_pt(str(graph_file))
        total_influence, cumsum_total_influence, non_selected_influence, cumsum_non_selected_influence, selected_influence, cumsum_selected_influence = compute_path_length_influence(graph, selected_nodes)
        
        # Add to appropriate lists
        cumsum_path_influences.append(cumsum_total_influence)
        path_influences.append(total_influence)
        non_selected_influences.append(non_selected_influence)
        cumsum_non_selected_influences.append(cumsum_non_selected_influence)
        selected_influences.append(selected_influence)
        cumsum_selected_influences.append(cumsum_selected_influence)
         
    cumsum_path_influences = torch.stack(cumsum_path_influences)
    path_influences = torch.stack(path_influences)
    non_selected_influences = torch.stack(non_selected_influences)
    cumsum_non_selected_influences = torch.stack(cumsum_non_selected_influences)
    selected_influences = torch.stack(selected_influences)
    cumsum_selected_influences = torch.stack(cumsum_selected_influences)
    
    # Create boolean masks for filtering
    is_a = metadata['correct_articles'] == 'a'
    is_an = metadata['correct_articles'] == 'an'
    # The 'correct?' column is assumed to exist and be boolean
    is_correct = metadata['correct?'] == True
    is_incorrect = ~is_correct

    def get_mean_influence(mask):
        """Calculate mean influence for each type of influence metric"""
        if not mask.any():
            # Return zeros if no examples match the mask
            return {
                'total': torch.zeros_like(cumsum_path_influences[0]),
                'non_selected': torch.zeros_like(cumsum_non_selected_influences[0]),
                'selected': torch.zeros_like(cumsum_selected_influences[0])
            }

        if isinstance(mask, pd.Series):
            mask = mask.values
        
        return {
            'total': cumsum_path_influences[mask].mean(0),
            'non_selected': cumsum_non_selected_influences[mask].mean(0),
            'selected': cumsum_selected_influences[mask].mean(0)
        }

    # Store results for this model, calculating means for each group
    all_model_results[model] = {
        'all': get_mean_influence(np.ones_like(is_a, dtype=bool)),
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
        'per_example_path_influences': path_influences,
        'per_example_non_selected_influences': non_selected_influences,
        'per_example_cumsum_non_selected_influences': cumsum_non_selected_influences,
        'per_example_selected_influences': selected_influences,
        'per_example_cumsum_selected_influences': cumsum_selected_influences
    }
    
    # Save results for this model
    PATH_LENGTH_RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(all_model_results[model], PATH_LENGTH_RESULTS_DIR / f'{logit_lens_model_name}_path_length_results.pt')
    print(f"  Saved results to {PATH_LENGTH_RESULTS_DIR / f'{logit_lens_model_name}_path_length_results.pt'}")
#%%
# Load results from disk for plotting
def load_saved_results(models):
    """Load saved results from disk"""
    loaded_results = {}
    
    for model in models:
        # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
        model_name_parts = model.split('-')
        size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
        if size_part.endswith('B'):
            size_part = size_part[:-1] + 'B'
        logit_lens_model_name = f"Qwen3-{size_part}"
        
        results_file = PATH_LENGTH_RESULTS_DIR / f'{logit_lens_model_name}_path_length_results.pt'
        loaded_results[model] = torch.load(results_file, weights_only=False)
        print(f"Loaded results for {model} from {results_file}")
    
    return loaded_results

# Load results from disk
all_model_results = load_saved_results(models)
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
#%%
# Plot selected vs non-selected path influences for each model
for model in models:
    clean_name = get_clean_model_name(model)
    results = all_model_results[model]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    x_range = range(1, len(results['all']['total']) + 1)
    
    # Plot for 'a' correct examples
    ax1.plot(x_range, results['a_correct']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax1.plot(x_range, results['a_correct']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax1.set_title(f'"a" Correct Examples')
    ax1.set_xlabel('Path Length')
    ax1.set_ylabel('Cumulative Path Influence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for 'a' incorrect examples
    ax2.plot(x_range, results['a_incorrect']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax2.plot(x_range, results['a_incorrect']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax2.set_title(f'"a" Incorrect Examples')
    ax2.set_xlabel('Path Length')
    ax2.set_ylabel('Cumulative Path Influence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot for 'an' correct examples
    ax3.plot(x_range, results['an_correct']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax3.plot(x_range, results['an_correct']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax3.set_title(f'"an" Correct Examples')
    ax3.set_xlabel('Path Length')
    ax3.set_ylabel('Cumulative Path Influence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot for 'an' incorrect examples
    ax4.plot(x_range, results['an_incorrect']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax4.plot(x_range, results['an_incorrect']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax4.set_title(f'"an" Incorrect Examples')
    ax4.set_xlabel('Path Length')
    ax4.set_ylabel('Cumulative Path Influence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Selected vs Non-selected Path Influences - {clean_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

#%%
# Plot relationship between selected node influence and model performance
def plot_selected_influence_vs_performance(model_results, model_name):
    """Create scatter plots for selected node influence vs performance"""
    metadata = model_results['metadata']
    cumsum_selected = model_results['per_example_cumsum_selected_influences']
    final_selected_influence = cumsum_selected[:, -1]  # Get the last cumsum value for each example
    
    # Get logit probabilities from metadata
    logit_probs = torch.tensor(metadata['logit_probability'].values)
    
    # Split by article type
    is_a = metadata['correct_articles'] == 'a'
    is_an = metadata['correct_articles'] == 'an'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for 'a' examples
    a_influences = final_selected_influence[is_a].cpu().numpy()
    a_probs = logit_probs[is_a].cpu().numpy()
    ax1.scatter(a_influences, a_probs, alpha=0.5)
    ax1.set_xlabel('Final Cumulative Selected Node Influence')
    ax1.set_ylabel('Probability of Correct Article')
    ax1.set_title(f'{model_name} - "a" Examples')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    a_corr = np.corrcoef(a_influences, a_probs)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {a_corr:.3f}', 
             transform=ax1.transAxes, verticalalignment='top')
    
    # Plot for 'an' examples
    an_influences = final_selected_influence[is_an].cpu().numpy()
    an_probs = logit_probs[is_an].cpu().numpy()
    ax2.scatter(an_influences, an_probs, alpha=0.5)
    ax2.set_xlabel('Final Cumulative Selected Node Influence')
    ax2.set_ylabel('Probability of Correct Article')
    ax2.set_title(f'{model_name} - "an" Examples')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    an_corr = np.corrcoef(an_influences, an_probs)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {an_corr:.3f}', 
             transform=ax2.transAxes, verticalalignment='top')
    
    plt.suptitle(f'Selected Node Influence vs Performance - {model_name}')
    plt.tight_layout()
    plt.show()
    
    return a_corr, an_corr

# # Plot for each model and collect correlations
# correlations = {}
# for model in models:
#     clean_name = get_clean_model_name(model)
#     a_corr, an_corr = plot_selected_influence_vs_performance(all_model_results[model], clean_name)
#     correlations[clean_name] = {'a': a_corr, 'an': an_corr}

# #%%
# # Plot correlation coefficients across models
# fig, ax = plt.subplots(figsize=(10, 6))

# x = np.arange(len(correlations))
# width = 0.35

# a_corrs = [corr['a'] for corr in correlations.values()]
# an_corrs = [corr['an'] for corr in correlations.values()]

# ax.bar(x - width/2, a_corrs, width, label='"a" examples')
# ax.bar(x + width/2, an_corrs, width, label='"an" examples')

# ax.set_ylabel('Correlation Coefficient')
# ax.set_title('Correlation between Selected Node Influence and Performance')
# ax.set_xticks(x)
# ax.set_xticklabels(list(correlations.keys()), rotation=45)
# ax.legend()
# ax.grid(True, alpha=0.3)

# plt.tight_layout()

#%%
# Create aggregated plots across models
def create_aggregated_plot(article_type, all_model_results, models, xlim=None, model_sizes=None, to_plot=['selected', 'non_selected']):
    """Create aggregated plot for a specific article type ('a' or 'an')"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define line styles for different categories
    line_styles = {
        ('selected', True): '-',      # solid
        ('selected', False): '--',    # dashed
        ('non_selected', True): ':',  # dotted
        ('non_selected', False): '-.' # dash-dot
    }
    
    # Define colors for each model
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']  # Classic distinct colors
    
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    for model_idx, model in enumerate(models):
        clean_name = get_clean_model_name(model)
        results = all_model_results[model]
        color = colors[model_idx]
        
        x_range = np.arange(1, len(results['all']['total']) + 1)
        if model_sizes:
            x_range = x_range / model_sizes[model_idx]
        
        # Plot selected and non-selected for both correct and incorrect cases
        for influence_type in to_plot:
            for is_correct in [True, False]:
                key = f"{article_type}_{'correct' if is_correct else 'incorrect'}"
                data = results[key][influence_type].cpu().numpy()
                
                line = ax.plot(x_range, data, 
                             line_styles[(influence_type, is_correct)],
                             color=color,
                             label=f"{clean_name} - {influence_type} ({'correct' if is_correct else 'incorrect'})",
                             linewidth=2)
                
                lines.append(line[0])
                labels.append(f"{clean_name} - {influence_type} ({'correct' if is_correct else 'incorrect'})")
    
    ax.set_xlabel('Path Length')
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylabel('Cumulative Path Influence')
    ax.set_title(f'Aggregated Path Influences - "{article_type}" Examples')
    ax.grid(True, alpha=0.3)
    
    # Create legend at the bottom
    ax.legend(lines, labels, 
             loc='upper center', 
             bbox_to_anchor=(0.5, -0.15),
             ncol=5,
             fontsize='small')
    
    plt.tight_layout()
    plt.show()

# Create separate plots for 'a' and 'an'
create_aggregated_plot('a', all_model_results, models)
create_aggregated_plot('an', all_model_results, models)
create_aggregated_plot('a', all_model_results, models, xlim=(0,12))
create_aggregated_plot('an', all_model_results, models, xlim=(0,12))

# %%
create_aggregated_plot('a', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes)
create_aggregated_plot('an', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes)
# %%
create_aggregated_plot('a', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes, to_plot=['selected'])
create_aggregated_plot('an', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes, to_plot=['selected'])
# %%
