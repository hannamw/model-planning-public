#%%
from pathlib import Path
from typing import List
from collections import namedtuple

import torch
import pandas as pd

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files

from utils import create_dataset_examples

Example = namedtuple("Example", ["sentence", "coninuation", "name"])

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

model_names_and_configs = [
    ('Qwen/Qwen3-0.6B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-0.6b-relu-lowl0.yaml'),
    ('Qwen/Qwen3-1.7B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-1.7b-relu-lowl0.yaml'),
    ('Qwen/Qwen3-4B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-4b-relu.yaml'),
    ('Qwen/Qwen3-8B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-8b-relu.yaml'),
    ('Qwen/Qwen3-14B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'),
    ]
batch_size = []


def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

df = pd.read_csv('data/professions_dataset_with_articles.csv')
df_ex = create_dataset_examples(df)


for model_name, model_config in model_names_and_configs:
    model_short_name = Path(model_config).stem
    print(model_short_name)
    model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)

    examples = [Example(f"{model.tokenizer.eos_token}{prompt}", f' {article}', f'{article}-{profession}') 
                for prompt, article, profession in zip(df_ex['Prompt'], df_ex['Article'], df_ex['Profession'])]

    for sentence, continuation, name in examples:
        input_ids = model.tokenizer(sentence).input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        with torch.inference_mode():
            logits = model(sentence)
            print_topk(model,logits)

        graph = attribute(sentence, model, batch_size=128, max_feature_nodes=7500, 
                        offload=None, verbose=True)

        pt_output_path = Path(f'attribution_graphs/{model_short_name}')
        pt_output_path.mkdir(exist_ok=True, parents=True)
        pt_output_path = pt_output_path / f'{name}.pt'
        graph.to_pt(pt_output_path)

        slug = f"{model_short_name}-{name}"

        json_output_path = Path(f'graph_files/{model_short_name}')
        json_output_path.mkdir(exist_ok=True, parents=True)
        json_output_path = json_output_path / f'{name}.pt'
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

    del model
    torch.cuda.empty_cache()
# %%
