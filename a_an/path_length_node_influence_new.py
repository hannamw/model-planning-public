#%%
# Import additional required modules
import re
import json
from pathlib import Path
from typing import List
from functools import partial, lru_cache

import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from circuit_tracer.graph import Graph, normalize_matrix
from load_feature_from_binary import get_features_top_acts_from_list

# Define threshold parameters

# Define models using same format as intervention file
models_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


def get_features_with_cache(features: list[tuple[int,int]], cache: dict, model_name: str, verbose=False):
    features_to_get = [feature for feature in features if feature not in cache]
    if features_to_get:
        new_features = get_features_top_acts_from_list(model_name, features_to_get, verbose=verbose)
        cache.update(new_features)
    return {feature: cache[feature] for feature in features if cache[feature] is not None}

def _is_word_feature(layer, feature_idx, word, feature_cache):
    feature_info = feature_cache[(layer, feature_idx)]
    if feature_info is None:
        return False
    word_counts = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        top_segment = ''.join(tokens[top_index - 10: top_index + 10])
        if word in top_segment:
            word_counts += 1
    
    in_logits = term_in_logits(word, feature_info['top_logits'], feature_info['bottom_logits'])
    return (word_counts > 5) or in_logits

def term_in_logits(term:str, top: list[str], bottom: list[str], use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    term = term.strip().lower()
    if substring_ok:
        len1 = 0
        for logit in logits:
            if logit == '' or logit == 'a' or logit == 'an':
                continue
            if term.startswith(logit):
                if len(logit) == 1:
                    len1 += 1
                    if len1 >= 2:
                        return True
                else:
                    return True
        return False
    else:
        return any(term in logit for logit in logits)



#%%
def load_model_results(model_name: str, results_dir: str = 'results/graph_metadata'):
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    metadata_path = results_dir / f'{model_name}.csv'
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
for model_name, transcoders in list(models_to_transcoders.items()):
    print(f"Processing model: {model_name}")
    
    feature_info_cache = {}
    is_word_feature = lru_cache(maxsize=None)(partial(_is_word_feature, feature_cache=feature_info_cache))
    
    metadata = load_model_results(model_name)
    
    graph_dir = Path('attribution_graphs') / model_name
    
    # Initialize lists for different article types
    cumsum_path_influences = []
    path_influences = []
    non_selected_influences = []
    cumsum_non_selected_influences = []
    selected_influences = []
    cumsum_selected_influences = []
    
    # Build feature set for caching
    feature_set = set()
    for idx, row in metadata.iterrows():
        article = row['article']
        planned = row['planned']
        filename = f'{article}-{planned}.pt'
        graph_file = graph_dir / filename
        graph = Graph.from_pt(graph_file)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        feature_set.update((layer, feature) for layer, _, feature in last_word_features.tolist())
    
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)
    
    # Process each example based on metadata
    for _, row in tqdm(metadata.iterrows()):
        article = row['article']
        planned = row['planned']
        
        # Generate filename based on metadata
        filename = f"{article}-{planned}.pt"
        graph_file = graph_dir / filename

        graph = Graph.from_pt(str(graph_file))
        
        # Get important nodes for this example - inline the logic
        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_nodes = [(layer, pos, feature) for layer, pos, feature in last_word_features.tolist() 
                            if is_word_feature(layer, feature, planned)]
        selected_nodes = selected_nodes if selected_nodes else None
    
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
    is_a = metadata['article'] == 'a'
    is_an = metadata['article'] == 'an'
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
    all_model_results[model_name] = {
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
    path_length_results_dir = Path('results/path_length')
    path_length_results_dir.mkdir(exist_ok=True)
    torch.save(all_model_results[model_name], path_length_results_dir / f'{model_name}_path_length_results.pt')
    print(f"  Saved results to {path_length_results_dir / f'{model_name}_path_length_results.pt'}")
