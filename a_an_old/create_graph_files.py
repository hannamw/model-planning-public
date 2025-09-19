#%%
from circuit_tracer.graph import Graph
from pathlib import Path 
from circuit_tracer.utils.create_graph_files import create_graph_files

base_path = Path('attribution_graphs_diff')

for model in ['Qwen3-0.6B','Qwen3-1.7B','Qwen3-4B']:
    for graph_file in (base_path / model).iterdir():
        graph = Graph.from_pt(graph_file)
        name = graph_file.stem
        print(name)
        slug = f"{model}-{name}"

        json_output_path = './graph_files_diff/'
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)
# %%
