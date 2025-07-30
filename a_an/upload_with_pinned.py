#%%
import json
from pathlib import Path
from typing import List
from collections import namedtuple
import torch
from urllib.parse import quote

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.upload_graph_to_s3 import upload_graph_to_s3
#%%
Example = namedtuple("Example", ["sentence", "coninuation", "name"])

models_to_configs = {
    'Qwen/Qwen3-0.6B': 'qwen3-0.6b-relu-lowl0',
    'Qwen/Qwen3-1.7B': 'qwen3-1.7b-relu-lowl0',
    'Qwen/Qwen3-4B': 'qwen3-4b-relu',
    'Qwen/Qwen3-8B': 'qwen3-8b-relu',
    'Qwen/Qwen3-14B': 'qwen3-14b-relu-lowl0',
}

model_name = "Qwen/Qwen3-8B"
lowercase_noslash = model_name.split('/')[-1].lower()
graph_name = 'an-archaeologist'
slug = f'{lowercase_noslash}-{graph_name}'

# Load relevant nodes
relevant_nodes_path = f'results/relevant_nodes/{model_name.split("/")[1]}/{graph_name}.json'
with open(relevant_nodes_path) as f:
    relevant_nodes = json.load(f)

required_profession_count = 5
required_related_terms_count = 10

# Filter nodes by profession and related terms
filtered_nodes = {}
for node_id, data in relevant_nodes['feature_counts'].items():
    if data['profession_count'] > required_profession_count or data['related_terms_count'] > required_related_terms_count:
        filtered_nodes[node_id] = data

print("Total nodes is", len(filtered_nodes))

# Sort nodes by total count
sorted_nodes = sorted(filtered_nodes.items(), key=lambda x: x[1]['total_count'], reverse=True)

# Format as clerps
pinned_ids = []
for node_id, data in sorted_nodes:
    layer, pos, feature_idx = node_id.split('_')
    node_key = f"{layer}_{feature_idx}_{pos}"
    pinned_ids.append(node_key)

output_path = f'attribution_graphs/{models_to_configs[model_name]}/{graph_name}.pt'

upload_graph_to_s3(output_path, slug, node_threshold=0.8, edge_threshold=0.95)
print(f"Graph now available at http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug={slug}&pinnedIds={'%2C'.join(pinned_ids)}")
# %%
