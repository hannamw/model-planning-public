#%%
# Import additional required modules
import re
from pathlib import Path
from functools import partial, lru_cache

import pandas as pd
from tqdm import tqdm
import torch

from circuit_tracer.graph import Graph
from circuit_tracer import ReplacementModel

from load_feature_from_binary import get_features_top_acts_from_list

# Define threshold parameters
models_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


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

#%%
# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
for model_name, transcoders in list(models_to_transcoders.items()):
    feature_info_cache = {}
    is_word_feature = lru_cache(maxsize=None)(partial(_is_word_feature, feature_cache=feature_info_cache))
    
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                            transcoders, dtype=torch.bfloat16,
                                            lazy_encoder=False)
    
    metadata = pd.read_csv(f'results/graph_metadata/{model_name}.csv')
    graph_dir = Path('attribution_graphs') / model_name
    
    # Add new columns to metadata for storing results
    metadata['original_a_prob'] = None
    metadata['original_an_prob'] = None
    
    # All intervention results (unconstrained)
    metadata['all_zeroed_a_prob'] = None
    metadata['all_zeroed_an_prob'] = None
    metadata['all_multiplied_a_prob'] = None
    metadata['all_multiplied_an_prob'] = None
    
    # Direct intervention results (constrained layers)
    metadata['direct_zeroed_a_prob'] = None
    metadata['direct_zeroed_an_prob'] = None
    metadata['direct_multiplied_a_prob'] = None
    metadata['direct_multiplied_an_prob'] = None
    
    # Random intervention results
    metadata['random_zeroed_a_prob'] = None
    metadata['random_zeroed_an_prob'] = None
    metadata['random_multiplied_a_prob'] = None
    metadata['random_multiplied_an_prob'] = None
    
    metadata['selected_nodes_count'] = None


    feature_set = set()
    for idx, row in metadata.iterrows():
        article = row['article']
        planned = row['planned']
        filename = f'{article}-{planned}.pt'
        #filename = row['filename']
        graph_file = graph_dir / filename
        graph = Graph.from_pt(graph_file)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        feature_set.update((layer, feature) for layer, _, feature in last_word_features.tolist())
    
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)

    # Process each example based on metadata
    for idx, row in tqdm(metadata.iterrows()):
        article = row['article']
        planned = row['planned']
        filename = f'{article}-{planned}.pt'
        #filename = row['filename']
        
        # Generate filename based on metadata
        graph_file = graph_dir / filename
        graph = Graph.from_pt(str(graph_file))

        # Filter nodes by profession and related terms
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_nodes = [(layer, pos, feature) for layer, pos, feature in last_word_features.tolist() 
                            if is_word_feature(layer, feature, planned)]
        
        s = graph.input_string
        
        original_logits, original_acts = model.get_activations(s)

        # Get token IDs for 'a' and 'an'
        tokenizer = model.tokenizer
        a_token_id = tokenizer.encode(' a', add_special_tokens=False)[0]
        an_token_id = tokenizer.encode(' an', add_special_tokens=False)[0]
        
        # Extract probabilities for 'a' and 'an' tokens
        # Apply softmax to get probabilities
        original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)
        
        # Get specific probabilities for 'a' and 'an'
        original_a_prob = original_probs[a_token_id].item()
        original_an_prob = original_probs[an_token_id].item()
        
        # If we have selected nodes, perform interventions
        if selected_nodes is not None and len(selected_nodes) > 0:
            selected_nodes_count = len(selected_nodes)
            
            # Prepare interventions for selected nodes
            zero_interventions = [(*feat, 0) for feat in selected_nodes]
            multiply_interventions = [(*feat, 5 * original_acts[tuple(feat)]) for feat in selected_nodes]
            
            # ALL intervention (unconstrained)
            logits_zeros_all, _ = model.feature_intervention(s, interventions=zero_interventions, return_activations=False)
            logits_multiply_all, _ = model.feature_intervention(s, interventions=multiply_interventions, return_activations=False)
            
            zerod_probs_all = torch.softmax(logits_zeros_all[0, -1, :], dim=-1)
            multiplied_probs_all = torch.softmax(logits_multiply_all[0, -1, :], dim=-1)
            
            all_zeroed_a_prob = zerod_probs_all[a_token_id].item()
            all_zeroed_an_prob = zerod_probs_all[an_token_id].item()
            all_multiplied_a_prob = multiplied_probs_all[a_token_id].item()
            all_multiplied_an_prob = multiplied_probs_all[an_token_id].item()

            # DIRECT intervention (constrained layers)
            logits_zeros_direct, _ = model.feature_intervention(s, interventions=zero_interventions, 
                                                                    constrained_layers=range(1, model.cfg.n_layers), 
                                                                    return_activations=False)
            logits_multiply_direct, _ = model.feature_intervention(s, interventions=multiply_interventions, 
                                                                        constrained_layers=range(1, model.cfg.n_layers), 
                                                                        return_activations=False)
            
            zerod_probs_direct = torch.softmax(logits_zeros_direct[0, -1, :], dim=-1)
            multiplied_probs_direct = torch.softmax(logits_multiply_direct[0, -1, :], dim=-1)
            
            direct_zeroed_a_prob = zerod_probs_direct[a_token_id].item()
            direct_zeroed_an_prob = zerod_probs_direct[an_token_id].item()
            direct_multiplied_a_prob = multiplied_probs_direct[a_token_id].item()
            direct_multiplied_an_prob = multiplied_probs_direct[an_token_id].item()

            # RANDOM intervention
            random_idxs = torch.randperm(len(last_word_features))[:len(selected_nodes)]
            random_nodes = last_word_features[random_idxs].tolist()
            zero_interventions_random = [(*feat, 0) for feat in random_nodes]
            multiply_interventions_random = [(*feat, 5 * original_acts[tuple(feat)]) for feat in random_nodes]
            logits_zeros_random, acts_zeros_random = model.feature_intervention(s, interventions=zero_interventions_random)
            logits_multiply_random, acts_multiply_random = model.feature_intervention(s, interventions=multiply_interventions_random)
            
            zerod_probs_random = torch.softmax(logits_zeros_random[0, -1, :], dim=-1)
            multiplied_probs_random = torch.softmax(logits_multiply_random[0, -1, :], dim=-1)
            
            random_zeroed_a_prob = zerod_probs_random[a_token_id].item()
            random_zeroed_an_prob = zerod_probs_random[an_token_id].item()
            random_multiplied_a_prob = multiplied_probs_random[a_token_id].item()
            random_multiplied_an_prob = multiplied_probs_random[an_token_id].item()
        else:
            # No selected nodes - use original probabilities for all interventions
            selected_nodes_count = 0
            all_zeroed_a_prob = original_a_prob
            all_zeroed_an_prob = original_an_prob
            all_multiplied_a_prob = original_a_prob
            all_multiplied_an_prob = original_an_prob
            direct_zeroed_a_prob = original_a_prob
            direct_zeroed_an_prob = original_an_prob
            direct_multiplied_a_prob = original_a_prob
            direct_multiplied_an_prob = original_an_prob
            random_zeroed_a_prob = original_a_prob
            random_zeroed_an_prob = original_an_prob
            random_multiplied_a_prob = original_a_prob
            random_multiplied_an_prob = original_an_prob
        
        # Store probabilities directly in metadata DataFrame
        metadata.at[idx, 'original_a_prob'] = original_a_prob
        metadata.at[idx, 'original_an_prob'] = original_an_prob
        
        # All intervention results
        metadata.at[idx, 'all_zeroed_a_prob'] = all_zeroed_a_prob
        metadata.at[idx, 'all_zeroed_an_prob'] = all_zeroed_an_prob
        metadata.at[idx, 'all_multiplied_a_prob'] = all_multiplied_a_prob
        metadata.at[idx, 'all_multiplied_an_prob'] = all_multiplied_an_prob
        
        # Direct intervention results
        metadata.at[idx, 'direct_zeroed_a_prob'] = direct_zeroed_a_prob
        metadata.at[idx, 'direct_zeroed_an_prob'] = direct_zeroed_an_prob
        metadata.at[idx, 'direct_multiplied_a_prob'] = direct_multiplied_a_prob
        metadata.at[idx, 'direct_multiplied_an_prob'] = direct_multiplied_an_prob
        
        # Random intervention results
        metadata.at[idx, 'random_zeroed_a_prob'] = random_zeroed_a_prob
        metadata.at[idx, 'random_zeroed_an_prob'] = random_zeroed_an_prob
        metadata.at[idx, 'random_multiplied_a_prob'] = random_multiplied_a_prob
        metadata.at[idx, 'random_multiplied_an_prob'] = random_multiplied_an_prob
        
        metadata.at[idx, 'selected_nodes_count'] = selected_nodes_count
        
    # Save the metadata with intervention results as CSV
    intervention_results_dir = Path('results/interventions')
    intervention_results_dir.mkdir(exist_ok=True)
    metadata.to_csv(intervention_results_dir / f'{model_name}.csv', index=False)
    print(f"  Saved intervention results to {intervention_results_dir / f'{model_name}.csv'}")
    
    # Count examples with and without important nodes
    examples_with_nodes = (metadata['selected_nodes_count'] > 0).sum()
    examples_without_nodes = (metadata['selected_nodes_count'] == 0).sum()
    
    print(f"  Processed {len(metadata)} total examples:")
    print(f"    - {examples_with_nodes} examples with important nodes (interventions performed)")
    print(f"    - {examples_without_nodes} examples without important nodes (original probabilities recorded)")
    del model
    torch.cuda.empty_cache()
