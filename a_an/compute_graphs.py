#%%
from pathlib import Path
from collections import namedtuple

import torch
import pandas as pd

from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files

Example = namedtuple("Example", ["sentence", "coninuation", "name"])

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

models_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}

for model_name, transcoders in models_to_transcoders.items():
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                            transcoders, 
                                            dtype=torch.bfloat16)
    df = pd.read_csv(f'results/behavioral/{model_name}.csv')
    
    # Create graph metadata file with slug column
    df_metadata = df.copy()
    df_metadata['slug'] = df_metadata.apply(lambda row: f"{model_name}-{row['article']}-{row['planned']}", axis=1)
    df_metadata['filename'] = df_metadata.apply(lambda row: f"{row['article']}-{row['planned']}.pt", axis=1)
    
    metadata_output_path = Path(f'results/graph_metadata')
    metadata_output_path.mkdir(exist_ok=True, parents=True)
    df_metadata.to_csv(metadata_output_path / f'{model_name}.csv', index=False)

    examples = [Example(prompt, f' {article}', f'{article}-{profession}') 
                for prompt, article, profession in zip(df['prompt_before_article'], df['article'], df['planned'])]

    for sentence, continuation, name in examples:
        input_ids = model.tokenizer(sentence).input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        with torch.inference_mode():
            logits = model(sentence)
            print_topk(model,logits)

        graph = attribute(sentence, model, batch_size=128, max_feature_nodes=7500, 
                        offload=None, verbose=True)

        pt_output_path = Path(f'attribution_graphs/{model_name}')
        pt_output_path.mkdir(exist_ok=True, parents=True)
        pt_output_path = pt_output_path / f'{name}.pt'
        graph.to_pt(pt_output_path)

        slug = f"{model_name}-{name}"

        json_output_path = Path(f'graph_files/{model_name}')
        json_output_path.mkdir(exist_ok=True, parents=True)
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

    del model
    torch.cuda.empty_cache()
