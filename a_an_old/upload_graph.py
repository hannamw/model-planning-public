#%%
from circuit_tracer.graph import Graph
from circuit_tracer.frontend.upload_graph_to_s3 import upload_graph_to_s3
#%%
graph = Graph.from_pt('attribution_graphs/qwen3-14b-relu-lowl0/a-banker.pt')
# %%
upload_graph_to_s3(graph, 'qwen3-14b-relu-lowl0-a-banker')
# %%
