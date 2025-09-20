#%%
# Import additional required modules
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torch

from circuit_tracer.graph import Graph
from circuit_tracer import ReplacementModel

# Define threshold parameters
INTERVENTION_RESULTS_DIR = Path('results/interventions')  # Directory for path length results
#%%

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

#%%
# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
for model_name, transcoders in models_to_transcoders.items():
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                                        transcoders, dtype=torch.bfloat16,
                                                        lazy_encoder=False)
    
    metadata = pd.read_csv('data/animals_dataset_downsampled.csv')
    
    graph_dir = Path('graphs_diff') / model_name
    
    # Add new columns to metadata for storing results
    metadata['original_are_prob'] = None
    metadata['original_is_prob'] = None
    metadata['zeroed_are_prob'] = None
    metadata['zeroed_is_prob'] = None
    metadata['multiplied_are_prob'] = None
    metadata['multiplied_is_prob'] = None
    metadata['selected_nodes_count'] = None
    
    # Process each example based on metadata
    for idx, row in tqdm(metadata.iterrows()):
        # Generate filename based on metadata
        relevant_terms = relevant_term_mapping[str(row['number'])]
        graph_name = row['name']
        filename = graph_name + ".pt"
        graph_file = graph_dir / filename
        
        graph = Graph.from_pt(str(graph_file))

        node_tensor = graph.active_features[graph.selected_features]
        node_list = node_tensor.tolist()
        
        # Filter nodes by profession and related terms
        selected_nodes = [(layer, pos, idx) for layer, pos, idx in node_list 
                        if any(term_in_logits(term, top_bottom_by_layer[layer][1][idx], 
                                            top_bottom_by_layer[layer][0][idx], k=5) for term in relevant_terms)]
        
        n_pos = graph.n_pos 
        if selected_nodes:
            selected_nodes = [(layer, pos, idx) for layer, pos, idx in selected_nodes if pos == n_pos - 1]
        s = graph.input_string
        
        original_logits, original_acts = model.get_activations(s)

        tokenizer = model.tokenizer
        are_token_id = tokenizer.encode(' are', add_special_tokens=False)[0]
        is_token_id = tokenizer.encode(' is', add_special_tokens=False)[0]
        
        # Extract probabilities for 'a' and 'an' tokens
        # Apply softmax to get probabilities
        original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)
        
        # Get specific probabilities for 'a' and 'an'
        original_are_prob = original_probs[are_token_id].item()
        original_is_prob = original_probs[is_token_id].item()
        
        # If we have selected nodes, perform interventions
        if selected_nodes is not None and len(selected_nodes) > 0:
            zero_interventions = [(*feat, 0) for feat in selected_nodes]
            multiply_interventions = [(*feat, 5 * original_acts[tuple(feat)]) for feat in selected_nodes]
            logits_zeros, acts_zeros = model.feature_intervention(s, interventions=zero_interventions)
            logits_multiply, acts_multiply = model.feature_intervention(s, interventions=multiply_interventions)
            
            zerod_probs = torch.softmax(logits_zeros[0, -1, :], dim=-1)
            multiplied_probs = torch.softmax(logits_multiply[0, -1, :], dim=-1)
            
            zeroed_are_prob = zerod_probs[are_token_id].item()
            zeroed_is_prob = zerod_probs[is_token_id].item()
            multiplied_are_prob = multiplied_probs[are_token_id].item()
            multiplied_is_prob = multiplied_probs[is_token_id].item()
            selected_nodes_count = len(selected_nodes)
        else:
            # No selected nodes - use original probabilities for interventions
            zeroed_are_prob = original_are_prob
            zeroed_is_prob = original_is_prob
            multiplied_are_prob = original_are_prob
            multiplied_is_prob = original_is_prob
            selected_nodes_count = 0
        
        # Store probabilities directly in metadata DataFrame
        metadata.at[idx, 'original_are_prob'] = original_are_prob
        metadata.at[idx, 'original_is_prob'] = original_is_prob
        metadata.at[idx, 'zeroed_are_prob'] = zeroed_are_prob
        metadata.at[idx, 'zeroed_is_prob'] = zeroed_is_prob
        metadata.at[idx, 'multiplied_are_prob'] = multiplied_are_prob
        metadata.at[idx, 'multiplied_is_prob'] = multiplied_is_prob
        metadata.at[idx, 'selected_nodes_count'] = selected_nodes_count
        
    # Save the metadata with intervention results as CSV
    INTERVENTION_RESULTS_DIR.mkdir(exist_ok=True)
    metadata.to_csv(INTERVENTION_RESULTS_DIR / f'{model_name}.csv', index=False)
    print(f"  Saved intervention results to {INTERVENTION_RESULTS_DIR / f'{model_name}.csv'}")
    
    # Count examples with and without important nodes
    examples_with_nodes = (metadata['selected_nodes_count'] > 0).sum()
    examples_without_nodes = (metadata['selected_nodes_count'] == 0).sum()
    
    print(f"  Processed {len(metadata)} total examples:")
    print(f"    - {examples_with_nodes} examples with important nodes (interventions performed)")
    print(f"    - {examples_without_nodes} examples without important nodes (original probabilities recorded)")

# %%
