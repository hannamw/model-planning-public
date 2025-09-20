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


model_sizes = [28,28, 36, 36, 40]

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

models_to_configs = {
    'Qwen3-0.6B': 'qwen3-0.6b-relu-lowl0',
    'Qwen3-1.7B': 'qwen3-1.7b-relu-lowl0',
    'Qwen3-4B': 'qwen3-4b-relu',
    'Qwen3-8B': 'qwen3-8b-relu',
    'Qwen3-14B': 'qwen3-14b-relu-lowl0',
}

models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']
model = models[-1]
top_bottom_by_layer = {}
print(f"Processing model: {model}")

#%%
# Load model metadata
# Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
model_name_parts = model.split('-')
size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
if size_part.endswith('B'):
    size_part = size_part[:-1] + 'B'
logit_lens_model_name = f"Qwen3-{size_part}"


metadata = pd.read_csv('data/animals_dataset_downsampled.csv')

graph_dir = Path('graphs_diff') / logit_lens_model_name

# Process each example based on metadata
for idx, row in tqdm(metadata.iterrows()):
    relevant_terms = relevant_term_mapping[str(row['number'])]
    # Generate filename based on metadata
    graph_name = row['name']
    filename = graph_name + ".pt"
    graph_file = graph_dir / filename
    
    graph = Graph.from_pt(str(graph_file))
    if not top_bottom_by_layer:
        print("Loading for the first time")
        for layer in range(graph.cfg.n_layers):
            top_bottom_by_layer[layer] = load_top_logits(logit_lens_model_name, layer)
        print("done")

    node_tensor = graph.active_features[graph.selected_features]
    node_list = node_tensor.tolist()


    # Filter nodes by profession and related terms
    selected_nodes = [(layer, pos, idx) for layer, pos, idx in node_list 
                        if any(term_in_logits(term, top_bottom_by_layer[layer][1][idx], 
                                            top_bottom_by_layer[layer][0][idx], k=5) for term in relevant_terms)]
    
    n_pos = graph.n_pos 
    #if LAST_ONLY and selected_nodes:
    #    selected_nodes = [(layer, pos, idx) for layer, pos, idx in selected_nodes if pos == n_pos - 1]
    
    # Format as clerps
    pinned_ids = []
    for node in selected_nodes:
        layer, pos, feature_idx = node
        node_key = f"{layer}_{feature_idx}_{pos}"
        if int(pos) == graph.n_pos - 1:
            pinned_ids.append(node_key)

    print(graph_name, "Total nodes is", len(selected_nodes), "filtered to", len(pinned_ids))

    slug = f'{logit_lens_model_name}-{graph_name}'
    x = f"localhost:8002/index.html?slug={slug}"
    if pinned_ids:
        x = x + f"&pinnedIds={'%2C'.join(pinned_ids[:100])}"
    print(x)
# %%
