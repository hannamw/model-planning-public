#%%
# Import additional required modules
import re
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
import torch

from circuit_tracer.graph import Graph
from circuit_tracer import ReplacementModel

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 1000

models_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


def load_top_logits(model_name, layer):
    with open(f'../cache/top_logits/{model_name}-{layer}.json', 'r') as f:
        return json.load(f)

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


def load_important_nodes(model_name: str, example_key: str, top_bottom_by_layer,
                        required_profession_count: int = 5,
                        required_related_terms_count: int = 10, k=10) -> List[List[int]]:
    """Load and filter important nodes based on profession and related terms counts
    
    Args:
        model_name: Name of the model (e.g. 'qwen3-0.6b-relu-lowl0')
        example_key: Example identifier (e.g. 'a-archaeologist')
        required_profession_count: Minimum profession count threshold
        required_related_terms_count: Minimum related terms count threshold
    
    Returns:
        List of [layer, feature, pos] lists for important nodes, or None if no nodes found
    """    
    # Load relevant nodes
    relevant_nodes_path = f'results/relevant_nodes_refined/{model_name}/{example_key}.json'
    with open(relevant_nodes_path) as f:
        relevant_nodes = json.load(f)

    article, profession = example_key.split('-')
    # Filter nodes by profession and related terms counts
    filtered_nodes = []
    for node_id, data in relevant_nodes['feature_counts'].items():
        layer, pos, feature = map(int, node_id.split('_'))
        
        top, bottom = top_bottom_by_layer[layer]
        top, bottom = top[feature], bottom[feature]
        if (data['profession_count'] > required_profession_count or 
            data['related_terms_count'] > required_related_terms_count or
            term_in_logits(profession, top, bottom, k=k)):
            # Convert node_id format from layer_pos_feature to [layer, feature, pos]

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


#%%
# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
for model_name, transcoders in models_to_transcoders.items():
    top_bottom_by_layer = {}
    
    # Load model metadata
    # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B

    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                                        transcoders, dtype=torch.bfloat16,
                                                        lazy_encoder=True)
    
    metadata = load_model_results(model_name)
    
    graph_dir = Path('attribution_graphs') / model_name
    
    # Add new columns to metadata for storing results
    metadata['original_a_prob'] = None
    metadata['original_an_prob'] = None
    metadata['zeroed_a_prob'] = None
    metadata['zeroed_an_prob'] = None
    metadata['multiplied_a_prob'] = None
    metadata['multiplied_an_prob'] = None
    metadata['selected_nodes_count'] = None
    
    # Process each example based on metadata
    for idx, row in tqdm(metadata.iterrows()):
        correct_article = row['correct_articles']
        profession = row['professions']
        
        # Generate filename based on metadata
        graph_name = f"{correct_article}-{profession}"
        filename = f"{correct_article}-{profession}.pt"
        graph_file = graph_dir / filename
        
        # Get important nodes for this example
        example_key = f"{correct_article}-{profession}"

        graph = Graph.from_pt(str(graph_file))
        if not top_bottom_by_layer:
            print("Loading for the first time")
            for layer in range(graph.cfg.n_layers):
                top_bottom_by_layer[layer] = load_top_logits(model_name, layer)
            print("done")

        # Filter nodes by profession and related terms
        selected_nodes = load_important_nodes(model_name, graph_name, top_bottom_by_layer, 
        REQUIRED_PROFESSION_COUNT, REQUIRED_RELATED_TERMS_COUNT, k=10)
        
        n_pos = graph.n_pos 
        if selected_nodes:
            selected_nodes = [(layer, pos, idx) for layer, pos, idx in selected_nodes if pos == n_pos - 1]
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
            zero_interventions = [(*feat, 0) for feat in selected_nodes]
            multiply_interventions = [(*feat, 5 * original_acts[tuple(feat)]) for feat in selected_nodes]
            logits_zeros, acts_zeros = model.feature_intervention(s, interventions=zero_interventions)
            logits_multiply, acts_multiply = model.feature_intervention(s, interventions=multiply_interventions)
            
            zerod_probs = torch.softmax(logits_zeros[0, -1, :], dim=-1)
            multiplied_probs = torch.softmax(logits_multiply[0, -1, :], dim=-1)
            
            zeroed_a_prob = zerod_probs[a_token_id].item()
            zeroed_an_prob = zerod_probs[an_token_id].item()
            multiplied_a_prob = multiplied_probs[a_token_id].item()
            multiplied_an_prob = multiplied_probs[an_token_id].item()
            selected_nodes_count = len(selected_nodes)
        else:
            # No selected nodes - use original probabilities for interventions
            zeroed_a_prob = original_a_prob
            zeroed_an_prob = original_an_prob
            multiplied_a_prob = original_a_prob
            multiplied_an_prob = original_an_prob
            selected_nodes_count = 0
        
        # Store probabilities directly in metadata DataFrame
        metadata.at[idx, 'original_a_prob'] = original_a_prob
        metadata.at[idx, 'original_an_prob'] = original_an_prob
        metadata.at[idx, 'zeroed_a_prob'] = zeroed_a_prob
        metadata.at[idx, 'zeroed_an_prob'] = zeroed_an_prob
        metadata.at[idx, 'multiplied_a_prob'] = multiplied_a_prob
        metadata.at[idx, 'multiplied_an_prob'] = multiplied_an_prob
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
