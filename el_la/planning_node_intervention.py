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
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


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
for model in models[-1:]:
    feature_info_cache = {}
    is_word_feature = lru_cache(maxsize=None)(partial(_is_word_feature, feature_cache=feature_info_cache))
    print(f"Processing model: {model}")
    
    # Load model metadata
    # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
    logit_lens_model_name = model

    replacement_model = ReplacementModel.from_pretrained('Qwen/' + logit_lens_model_name, 
                                                        models_and_transcoders['Qwen/' + logit_lens_model_name], dtype=torch.bfloat16,
                                                        cpu_encoder=False)
    
    metadata = pd.read_csv(f'results/behavioral/{model}.csv').head(150)
    
    graph_dir = Path('attribution_graphs') / model
    
    # Add new columns to metadata for storing results
    metadata['original_el_prob'] = None
    metadata['original_la_prob'] = None
    metadata['original_los_prob'] = None
    metadata['original_las_prob'] = None
    metadata['zeroed_el_prob'] = None
    metadata['zeroed_la_prob'] = None
    metadata['zeroed_los_prob'] = None
    metadata['zeroed_las_prob'] = None
    metadata['multiplied_el_prob'] = None
    metadata['multiplied_la_prob'] = None
    metadata['multiplied_los_prob'] = None
    metadata['multiplied_las_prob'] = None
    metadata['selected_nodes_count'] = None

    feature_set = set()
    for idx, row in metadata.iterrows():
        correct_article = row['article']
        noun = row['spanish_noun']
        
        # Generate filename based on metadata
        graph_name = f"{correct_article}-{noun}"
        filename = f"{correct_article}-{noun}.pt"
        graph_file = graph_dir / filename
        graph = Graph.from_pt(graph_file)

        input_tokens = replacement_model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(replacement_model.tokenizer.eos_token)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        feature_set.update((layer, feature) for layer, _, feature in last_word_features.tolist())

    # get all features at once
    get_features_with_cache(list(feature_set), feature_info_cache, model)
    
    # Process each example based on metadata
    for idx, row in tqdm(metadata.iterrows()):
        correct_article = row['article']
        noun = row['spanish_noun']
        
        # Generate filename based on metadata
        graph_name = f"{correct_article}-{noun}"
        filename = f"{correct_article}-{noun}.pt"
        graph_file = graph_dir / filename
        
        # Get important nodes for this example
        example_key = f"{correct_article}-{noun}"

        graph = Graph.from_pt(str(graph_file))

        # Filter nodes by profession and related terms
        input_tokens = replacement_model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(replacement_model.tokenizer.eos_token)

        selected_features = graph.active_features[graph.selected_features]
        last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_nodes = [(layer, pos, feature) for layer, pos, feature in last_word_features.tolist() 
                            if is_word_feature(layer, feature, noun)]
        
        n_pos = graph.n_pos
        s = graph.input_string
        
        original_logits, original_acts = replacement_model.get_activations(s)

        # Get token IDs for 'el', 'la', 'los', and 'las'
        tokenizer = replacement_model.tokenizer
        el_token_id = tokenizer.encode(' el', add_special_tokens=False)[0]
        la_token_id = tokenizer.encode(' la', add_special_tokens=False)[0]
        los_token_id = tokenizer.encode(' los', add_special_tokens=False)[0]
        las_token_id = tokenizer.encode(' las', add_special_tokens=False)[0]
        
        # Extract probabilities for 'el', 'la', 'los', and 'las' tokens
        # Apply softmax to get probabilities
        original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)
        
        # Get specific probabilities for 'el', 'la', 'los', and 'las'
        original_el_prob = original_probs[el_token_id].item()
        original_la_prob = original_probs[la_token_id].item()
        original_los_prob = original_probs[los_token_id].item()
        original_las_prob = original_probs[las_token_id].item()

        selected_nodes_count = len(selected_nodes)
        
        # If we have selected nodes, perform interventions
        if selected_nodes is not None and len(selected_nodes) > 0:
            #print(original_acts.size())
            #sn = torch.tensor(selected_nodes)
            #print(sn.max(0).values)
            zero_interventions = [(*feat, 0) for feat in selected_nodes]
            multiply_interventions = [(*feat, 5 * original_acts[tuple(feat)]) for feat in selected_nodes]
            logits_zeros, acts_zeros = replacement_model.feature_intervention(s, interventions=zero_interventions, return_activations=False)
            logits_multiply, acts_multiply = replacement_model.feature_intervention(s, interventions=multiply_interventions, return_activations=False)
            
            zerod_probs = torch.softmax(logits_zeros[0, -1, :], dim=-1)
            multiplied_probs = torch.softmax(logits_multiply[0, -1, :], dim=-1)
            
            zeroed_el_prob = zerod_probs[el_token_id].item()
            zeroed_la_prob = zerod_probs[la_token_id].item()
            zeroed_los_prob = zerod_probs[los_token_id].item()
            zeroed_las_prob = zerod_probs[las_token_id].item()
            multiplied_el_prob = multiplied_probs[el_token_id].item()
            multiplied_la_prob = multiplied_probs[la_token_id].item()
            multiplied_los_prob = multiplied_probs[los_token_id].item()
            multiplied_las_prob = multiplied_probs[las_token_id].item()
            
        else:
            # No selected nodes - use original probabilities for interventions
            zeroed_el_prob = original_el_prob
            zeroed_la_prob = original_la_prob
            zeroed_los_prob = original_los_prob
            zeroed_las_prob = original_las_prob
            multiplied_el_prob = original_el_prob
            multiplied_la_prob = original_la_prob
            multiplied_los_prob = original_los_prob
            multiplied_las_prob = original_las_prob
        
        # Store probabilities directly in metadata DataFrame
        metadata.at[idx, 'original_el_prob'] = original_el_prob
        metadata.at[idx, 'original_la_prob'] = original_la_prob
        metadata.at[idx, 'original_los_prob'] = original_los_prob
        metadata.at[idx, 'original_las_prob'] = original_las_prob
        metadata.at[idx, 'zeroed_el_prob'] = zeroed_el_prob
        metadata.at[idx, 'zeroed_la_prob'] = zeroed_la_prob
        metadata.at[idx, 'zeroed_los_prob'] = zeroed_los_prob
        metadata.at[idx, 'zeroed_las_prob'] = zeroed_las_prob
        metadata.at[idx, 'multiplied_el_prob'] = multiplied_el_prob
        metadata.at[idx, 'multiplied_la_prob'] = multiplied_la_prob
        metadata.at[idx, 'multiplied_los_prob'] = multiplied_los_prob
        metadata.at[idx, 'multiplied_las_prob'] = multiplied_las_prob
        metadata.at[idx, 'selected_nodes_count'] = selected_nodes_count
        
    # Save the metadata with intervention results as CSV
    intervention_results_dir = Path('results/interventions_last')
    intervention_results_dir.mkdir(exist_ok=True)
    metadata.to_csv(intervention_results_dir / f'{logit_lens_model_name}.csv', index=False)
    print(f"  Saved intervention results to {intervention_results_dir / f'{logit_lens_model_name}.csv'}")
    
    # Count examples with and without important nodes
    examples_with_nodes = (metadata['selected_nodes_count'] > 0).sum()
    examples_without_nodes = (metadata['selected_nodes_count'] == 0).sum()
    
    print(f"  Processed {len(metadata)} total examples:")
    print(f"    - {examples_with_nodes} examples with important nodes (interventions performed)")
    print(f"    - {examples_without_nodes} examples without important nodes (original probabilities recorded)")
    del model
    torch.cuda.empty_cache()

