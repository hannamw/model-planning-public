#%%
import json
from pathlib import Path
from typing import List
from collections import namedtuple

from circuit_tracer.graph import Graph
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.upload_graph_to_s3 import upload_graph_to_s3

#%%
def load_important_nodes(model_name: str, example_key: str, 
                        required_profession_count: int = 5,
                        required_related_terms_count: int = 10) -> List[List[int]]:
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


    # Filter nodes by profession and related terms counts
    filtered_nodes = []
    for node_id, data in relevant_nodes['feature_counts'].items():
        if (data['profession_count'] > required_profession_count or 
            data['related_terms_count'] > required_related_terms_count):
            # Convert node_id format from layer_pos_feature to [layer, feature, pos]
            layer, pos, feature = map(int, node_id.split('_'))
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
lowercase_noslash = model_name.split('/')[-1].lower()



graph_name = 'an-economist'
slug = f'{lowercase_noslash}-{graph_name}'

# Load relevant nodes
relevant_nodes_path = f'results/relevant_nodes_refined/{model_name.split("/")[1]}/{graph_name}.json'
with open(relevant_nodes_path) as f:
    relevant_nodes = json.load(f)

required_profession_count = 2
required_related_terms_count = 100

# Filter nodes by profession and related terms
filtered_nodes = {}
for node_id, data in relevant_nodes['feature_counts'].items():
    if data['profession_count'] > required_profession_count or data['related_terms_count'] > required_related_terms_count:
        filtered_nodes[node_id] = data

# Sort nodes by total count
sorted_nodes = sorted(filtered_nodes.items(), key=lambda x: x[1]['total_count'], reverse=True)

output_path = f'attribution_graphs/{models_to_configs[model_name]}/{graph_name}.pt'
graph = Graph.from_pt(output_path)

# Format as clerps
pinned_ids = []
for node_id, data in sorted_nodes:
    layer, pos, feature_idx = node_id.split('_')
    node_key = f"{layer}_{feature_idx}_{pos}"
    if int(pos) == graph.n_pos - 1:
        pinned_ids.append(node_key)

print("Total nodes is", len(filtered_nodes), "filtered to", len(pinned_ids))
#%%
upload_graph_to_s3(output_path, slug, node_threshold=0.8, edge_threshold=0.95)
print(f"Graph now available at http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug={slug}&pinnedIds={'%2C'.join(pinned_ids)}")
# %%
