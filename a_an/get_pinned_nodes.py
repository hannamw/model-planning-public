#%%
import re
import json
from pathlib import Path
from typing import List
from collections import namedtuple
from tqdm import tqdm

from circuit_tracer.graph import Graph
#%%
models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"}


def load_top_logits(model_name, layer):
    target = models_and_transcoders[model_name].split("/")[-1]
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
    relevant_nodes_path = f'results/relevant_nodes_refined/{model_name.split("/")[-1]}/{example_key}.json'
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
Example = namedtuple("Example", ["sentence", "coninuation", "name"])

models_to_configs = {
    'Qwen/Qwen3-0.6B': 'qwen3-0.6b-relu-lowl0',
    'Qwen/Qwen3-1.7B': 'qwen3-1.7b-relu-lowl0',
    'Qwen/Qwen3-4B': 'qwen3-4b-relu',
    'Qwen/Qwen3-8B': 'qwen3-8b-relu',
    'Qwen/Qwen3-14B': 'qwen3-14b-relu-lowl0',
}

model_name = "Qwen/Qwen3-14B"
model_name_noslash = model_name.split('/')[-1]
lowercase_noslash = model_name_noslash.lower()
import pandas as pd
df = pd.read_csv('results/logit-lens/Qwen3-14B/metadata.csv')
graph_names = [f'{article}-{profession}' for article, profession in zip(df['correct_articles'], df['professions'])]
top_bottom_by_layer = {}
#%%
for graph_name in graph_names:
    slug = f'{lowercase_noslash}-{graph_name}'

    required_profession_count = 5
    required_related_terms_count = 1000
    # Sort nodes by total count
    #sorted_nodes = sorted(filtered_nodes.items(), key=lambda x: x[1]['total_count'], reverse=True)

    output_path = f'attribution_graphs/{models_to_configs[model_name]}/{graph_name}.pt'
    graph = Graph.from_pt(output_path)

    if not top_bottom_by_layer:
        print("Loading for the first time")
        for layer in range(graph.cfg.n_layers):
            top_bottom_by_layer[layer] = load_top_logits(model_name, layer)
        print("done")

    # Filter nodes by profession and related terms
    filtered_nodes = load_important_nodes(model_name, graph_name, top_bottom_by_layer, 
        required_profession_count, required_related_terms_count, k=5)

    # Format as clerps
    pinned_ids = []
    for node in filtered_nodes:
        layer, pos, feature_idx = node
        node_key = f"{layer}_{feature_idx}_{pos}"
        if int(pos) == graph.n_pos - 1:
            pinned_ids.append(node_key)

    print(graph_name, "Total nodes is", len(filtered_nodes), "filtered to", len(pinned_ids))

    slug = f'{models_to_configs[model_name]}-{graph_name}'
    x = f"localhost:8002/index.html?slug={slug}"
    if pinned_ids:
        x = x + f"&pinnedIds={'%2C'.join(pinned_ids[:100])}"
    print(x)
# %%
