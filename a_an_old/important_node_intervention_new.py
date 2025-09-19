#%%
# Import additional required modules
import re
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from circuit_tracer.graph import Graph, normalize_matrix
from circuit_tracer import ReplacementModel

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 1000
LAST_ONLY = True
INTERVENTION_RESULTS_DIR = Path('results/interventions_last') if LAST_ONLY else Path('results/interventions')  # Directory for path length results

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

logit_lens_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


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
model = models[-1]
top_bottom_by_layer = {}
print(f"Processing model: {model}")

# Load model metadata
# Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
model_name_parts = model.split('-')
size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
if size_part.endswith('B'):
    size_part = size_part[:-1] + 'B'
logit_lens_model_name = f"Qwen3-{size_part}"

replacement_model = ReplacementModel.from_pretrained('Qwen/' + logit_lens_model_name, 
                                                    logit_lens_to_transcoders[logit_lens_model_name], dtype=torch.bfloat16,
                                                    lazy_encoder=True)

metadata = load_model_results(logit_lens_model_name)

graph_dir = Path('attribution_graphs') / model

#%%
profession = 'economist'
row = metadata[metadata['professions'] == profession].iloc[0]
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
        top_bottom_by_layer[layer] = load_top_logits(logit_lens_model_name, layer)
    print("done")

# Filter nodes by profession and related terms
selected_nodes = load_important_nodes(logit_lens_model_name, graph_name, top_bottom_by_layer, 
REQUIRED_PROFESSION_COUNT, REQUIRED_RELATED_TERMS_COUNT, k=10)

n_pos = graph.n_pos 
if LAST_ONLY and selected_nodes:
    selected_nodes = [(layer, pos, idx) for layer, pos, idx in selected_nodes if pos == n_pos - 1]
s = graph.input_string

original_logits, original_acts = replacement_model.get_activations(s)

# Get token IDs for 'a' and 'an'
tokenizer = replacement_model.tokenizer
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
    #print(original_acts.size())
    #sn = torch.tensor(selected_nodes)
    #print(sn.max(0).values)
    zero_interventions = [(*feat, 0) for feat in selected_nodes]
    multiply_interventions = [(*feat, 5 * original_acts[tuple(feat)]) for feat in selected_nodes]
    logits_zeros, acts_zeros = replacement_model.feature_intervention(s, interventions=zero_interventions)
    logits_multiply, acts_multiply = replacement_model.feature_intervention(s, interventions=multiply_interventions)
    
    zerod_probs = torch.softmax(logits_zeros[0, -1, :], dim=-1)
    multiplied_probs = torch.softmax(logits_multiply[0, -1, :], dim=-1)
    
    zeroed_a_prob = zerod_probs[a_token_id].item()
    zeroed_an_prob = zerod_probs[an_token_id].item()
    multiplied_a_prob = multiplied_probs[a_token_id].item()
    multiplied_an_prob = multiplied_probs[an_token_id].item()
    selected_nodes_count = len(selected_nodes)
    print(original_a_prob, original_an_prob)
    print(zeroed_a_prob, zeroed_an_prob)
    print(multiplied_a_prob, multiplied_an_prob)
else:
    # No selected nodes - use original probabilities for interventions
    zeroed_a_prob = original_a_prob
    zeroed_an_prob = original_an_prob
    multiplied_a_prob = original_a_prob
    multiplied_an_prob = original_an_prob
    selected_nodes_count = 0
    print("No nodes")
# %%
