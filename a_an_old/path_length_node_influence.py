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
REQUIRED_RELATED_TERMS_COUNT = 1000
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
        non_selected_influence_by_path_length[i] = current_paths.dot(logit_weights)
        selected_influence_by_path_length[i] = selected_paths.dot(logit_weights)
        if selected_nodes is not None:
            selected_paths += current_paths * selected_mask
            current_paths *= (1 - selected_mask)
        current_paths = A @ current_paths
        selected_paths = A @ selected_paths
        if not current_paths.any():
            break
        
        
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
