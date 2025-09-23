#%%
# Import additional required modules
import re
from pathlib import Path
from functools import partial, lru_cache

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from circuit_tracer.graph import Graph, normalize_matrix
from circuit_tracer import ReplacementModel
from load_feature_from_binary import get_features_top_acts_from_list

# Define models and directory
PATH_LENGTH_RESULTS_DIR = Path('results/path_length')  # Directory for path length results

models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]
models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
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
            if logit in {'', 'el', 'la', 'los', 'las', 'a', 'an'}:
                continue
            if term.startswith(logit):
                if len(logit) <= 2:
                    len1 += 1
                    if len1 >= 2:
                        return True
                else:
                    return True
        return False
    else:
        return any(term in logit for logit in logits)

#%%
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

# Load important nodes for all models using feature-based approach
# Features are loaded and cached, then filtered based on word content
for model_name in models:
    feature_info_cache = {}
    is_word_feature = lru_cache(maxsize=None)(partial(_is_word_feature, feature_cache=feature_info_cache))
    print(f"Processing model: {model_name}")
    
    # Load model metadata
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                                        models_and_transcoders['Qwen/' + model_name], dtype=torch.bfloat16,
                                                        cpu_encoder=False)
    
    metadata = pd.read_csv(f'results/behavioral/{model_name}.csv').head(150)
    
    graph_dir = Path('attribution_graphs') / model_name
    
    # Initialize lists for different article types
    cumsum_path_influences = []
    path_influences = []
    non_selected_influences = []
    cumsum_non_selected_influences = []
    selected_influences = []
    cumsum_selected_influences = []
    
    feature_set = set()
    for idx, row in metadata.iterrows():
        correct_article = row['article']
        noun = row['spanish_noun']
        
        # Generate filename based on metadata
        graph_name = f"{correct_article}-{noun}"
        filename = f"{correct_article}-{noun}.pt"
        graph_file = graph_dir / filename
        graph = Graph.from_pt(graph_file)

        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        feature_set.update((layer, feature) for layer, _, feature in last_word_features.tolist())

    # get all features at once
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)
    
    # Process each example based on metadata
    for idx, row in tqdm(metadata.iterrows()):
        correct_article = row['article']
        noun = row['spanish_noun']
        english_noun = row['english_noun']
        
        # Generate filename based on metadata
        graph_name = f"{correct_article}-{noun}"
        filename = f"{correct_article}-{noun}.pt"
        graph_file = graph_dir / filename
        
        # Get important nodes for this example
        example_key = f"{correct_article}-{noun}"

        graph = Graph.from_pt(str(graph_file))

        # Filter nodes by profession and related terms
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_nodes = [(layer, pos, feature) for layer, pos, feature in last_word_features.tolist() 
                            if is_word_feature(layer, feature, noun) or is_word_feature(layer, feature, english_noun)]
    
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
    is_el = metadata['article'] == 'el'
    is_la = metadata['article'] == 'la'
    is_los = metadata['article'] == 'los'
    is_las = metadata['article'] == 'las'
    # The 'correct?' column is assumed to exist and be boolean
    is_correct = metadata['article_correct'] == True
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
        'all': get_mean_influence(np.ones_like(is_el, dtype=bool)),
        'el': get_mean_influence(is_el),
        'la': get_mean_influence(is_la),
        'los': get_mean_influence(is_los),
        'las': get_mean_influence(is_las),
        'correct': get_mean_influence(is_correct),
        'incorrect': get_mean_influence(is_incorrect),
        'el_correct': get_mean_influence(is_el & is_correct),
        'el_incorrect': get_mean_influence(is_el & is_incorrect),
        'la_correct': get_mean_influence(is_la & is_correct),
        'la_incorrect': get_mean_influence(is_la & is_incorrect),
        'los_correct': get_mean_influence(is_los & is_correct),
        'los_incorrect': get_mean_influence(is_los & is_incorrect),
        'las_correct': get_mean_influence(is_las & is_correct),
        'las_incorrect': get_mean_influence(is_las & is_incorrect),
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
    torch.save(all_model_results[model], PATH_LENGTH_RESULTS_DIR / f'{model_name}.pt')
    print(f"  Saved results to {PATH_LENGTH_RESULTS_DIR / f'{model_name}.pt'}")
    del model
    torch.cuda.empty_cache()
