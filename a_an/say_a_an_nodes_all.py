#%%
# Import additional required modules
import re
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from circuit_tracer.graph import Graph

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 1000
LAST_ONLY = True
PATH_LENGTH_RESULTS_DIR = Path('results/directness_diff_last') if LAST_ONLY else Path('results/directness_diff')  # Directory for path length results

model_sizes = [28,28, 36, 36, 40]

logit_lens_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
models = list(logit_lens_to_transcoders.keys())

#%%
def load_top_logits(model_name, layer):
    target = logit_lens_to_transcoders[model_name].split("/")[-1]
    with open(f'../cache/top_logits/{target}-{layer}.json', 'r') as f:
        return json.load(f)

def term_in_logits(term:str, top: list[str], bottom: list[str], use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    term = term.strip().lower()
    if substring_ok:
        len1 = 0
        for logit in logits:
            if logit == '' or ((logit == 'a' or logit == 'an') and not (term == 'a' or term == 'an')):
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


def a_an_in_logits_count(top: list[str], bottom: list[str], count=2, use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    return sum((logit == 'a' or logit == 'an') for logit in logits) > count

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
# Store results for all models
all_model_results = {}

# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
for model in models:
    print(f"Processing model: {model}")
    
    # Load model metadata
    metadata = load_model_results(model)
    top_bottom_by_layer = {}
    
    graph_dir = Path('attribution_graphs_diff') / model
    
    # Initialize lists for different article types
    a_an_counts = []
    a_an_selected_counts = []
    
    # Process each example based on metadata
    for _, row in tqdm(metadata.iterrows()):
        correct_article = row['correct_articles']
        profession = row['professions']
        
        # Generate filename based on metadata
        filename = f"{correct_article}-{profession}.pt"
        graph_file = graph_dir / filename

        graph = Graph.from_pt(str(graph_file))

        if not top_bottom_by_layer:
            print("Loading for the first time")
            for layer in range(graph.cfg.n_layers):
                top_bottom_by_layer[layer] = load_top_logits(model, layer)
            print("done")
        
        # Get important nodes for this example
        example_key = f"{correct_article}-{profession}"

        a_an_mask = []
        for layer, pos, idx in graph.active_features.tolist():
            top, bottom = top_bottom_by_layer[layer]
            top, bottom = top[idx], bottom[idx]
            is_a_an = a_an_in_logits_count(top, bottom, count=3)
            a_an_mask.append(is_a_an)
        a_an_mask = torch.tensor(a_an_mask)
        a_an_count = a_an_mask.sum().item()
        selected_a_an_count = a_an_mask[graph.selected_features].sum().item()
        a_an_counts.append(a_an_count)
        a_an_selected_counts.append(selected_a_an_count)

    a_an_counts = torch.stack(a_an_counts)
    a_an_selected_counts = torch.stack(a_an_selected_counts)

    metadata['a_an_counts'] = a_an_counts
    metadata['a_an_selected_counts'] = a_an_selected_counts
    metadata.to_csv(f'results/a_an_nodes/{model}.csv', index=False)

# %%
