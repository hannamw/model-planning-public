#%%
# Import additional required modules
from pathlib import Path
from functools import partial, lru_cache

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from circuit_tracer.graph import Graph, normalize_matrix
from circuit_tracer import ReplacementModel

from load_feature_from_binary import get_features_top_acts_from_list

# Define threshold parameters

PATH_LENGTH_RESULTS_DIR = Path('results/path_length')  # Directory for path length results

models_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


relevant_term_mapping = {
    '1': ['¹', '₁', '①', '一', '١', 'one', '1', '１'],
    '2': ['²', '₂', '②', '二', '٢', 'two', '2', '２'],
    '3': ['³', '₃', '③', '三', '٣', 'three', '3', '３'],
    '4': ['⁴', '₄', '④', '四', '٤', 'four', '4', '４'],
    '5': ['⁵', '₅', '⑤', '五', '٥', 'five', '5', '５'],
    '6': ['⁶', '₆', '⑥', '六', '٦', 'six', '6', '６'],
    '7': ['⁷', '₇', '⑦', '七', '٧', 'seven', '7', '７'],
    '8': ['⁸', '₈', '⑧', '八', '٨', 'eight', '8', '８'],
    '9': ['⁹', '₉', '⑨', '九', '٩', 'nine', '9', '９'],
    '0': ['⁰', '₀', '⓪', '零', '٠', 'zero', '0', '０'],
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
    return term_in_logits(word, feature_info['top_logits'], feature_info['bottom_logits'])

def term_in_logits(term:str, top: list[str], bottom: list[str], use_bottom=False, substring_ok=False, k=10):
    logits = top[-k:] + bottom[:k] if use_bottom else top[-k:]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [logit.strip().lower() for logit in logits]
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
        return any(term == logit for logit in logits)


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
    feature_info_cache = {}
    is_word_feature = lru_cache(maxsize=None)(partial(_is_word_feature, feature_cache=feature_info_cache))
    
    print(f"Processing model: {model_name}")
    
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                            transcoders, dtype=torch.bfloat16,
                                            lazy_encoder=False)
    
    metadata = pd.read_csv('data/animals_dataset_downsampled.csv')
    
    graph_dir = Path('attribution_graphs') / model_name
    
    # Pre-load all features for this model
    feature_set = set()
    for _, row in metadata.iterrows():
        relevant_terms = relevant_term_mapping[str(row['number'])]
        graph_name = row['name']
        filename = graph_name + ".pt"
        graph_file = graph_dir / filename
        graph = Graph.from_pt(graph_file)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        feature_set.update((layer, feature) for layer, _, feature in last_word_features.tolist())
    
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)
    
    # Initialize lists for different article types
    cumsum_path_influences = []
    path_influences = []
    non_selected_influences = []
    cumsum_non_selected_influences = []
    selected_influences = []
    cumsum_selected_influences = []
    
    # Process each example based on metadata
    for _, row in tqdm(metadata.iterrows()):
        relevant_terms = relevant_term_mapping[str(row['number'])]
        graph_name = row['name']
        filename = graph_name + ".pt"
        graph_file = graph_dir / filename

        graph = Graph.from_pt(str(graph_file))
        
        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_nodes = [(layer, pos, feature) for layer, pos, feature in last_word_features.tolist() 
                            if any(is_word_feature(layer, feature, term) for term in relevant_terms)]
    
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
    are_mask = metadata['answer'] == 'are'
    is_mask = metadata['answer'] == 'is'

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
        'all': get_mean_influence(np.ones_like(are_mask, dtype=bool)),
        'are': get_mean_influence(are_mask),
        'is': get_mean_influence(is_mask),
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
    torch.save(all_model_results[model_name], PATH_LENGTH_RESULTS_DIR / f'{model_name}_path_length_results.pt')
    print(f"  Saved results to {PATH_LENGTH_RESULTS_DIR / f'{model_name}_path_length_results.pt'}")
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()

# %%
