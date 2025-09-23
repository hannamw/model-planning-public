#%%
from pathlib import Path
from typing import List
from collections import namedtuple

import torch
import pandas as pd

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files

Example = namedtuple("Example", ["sentence", "coninuation", "name"])

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

model_names_to_transcoders = {
    'Qwen3-0.6B': 'mwhanna/qwen3-0.6b-transcoders-lowl0',
    'Qwen3-1.7B': 'mwhanna/qwen3-1.7b-transcoders-lowl0',
    'Qwen3-4B': 'mwhanna/qwen3-4b-transcoders',
    'Qwen3-8B': 'mwhanna/qwen3-8b-transcoders',
    'Qwen3-14B': 'mwhanna/qwen3-14b-transcoders-lowl0',
}


def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified


for model_name, transcoders in model_names_to_transcoders.items():
    print(model_name)
    df = pd.read_csv(f'results/behavioral/{model_name}.csv').head(150)
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                            transcoders, 
                                            lazy_encoder=False,
                                            dtype=torch.bfloat16)

    el_token = model.tokenizer(' el').input_ids[0]
    la_token = model.tokenizer(' la').input_ids[0]
    los_token = model.tokenizer(' los').input_ids[0]
    las_token = model.tokenizer(' las').input_ids[0]

    el_minus_la = model.W_U[:, el_token] - model.W_U[:, la_token]
    la_minus_el = model.W_U[:, la_token] - model.W_U[:, el_token]
    los_minus_las = model.W_U[:, los_token] - model.W_U[:, las_token]
    las_minus_los = model.W_U[:, las_token] - model.W_U[:, los_token]

    examples = [Example(f"{model.tokenizer.eos_token}{prompt}", f' {article}', f'{article}-{profession}') 
                for prompt, article, profession in zip(df['prompt_before_article'], df['article'], df['spanish_noun'])]

    names = []
    for sentence, continuation, name in examples:
        input_ids = model.tokenizer(sentence).input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)

        if continuation == ' el':
            quantity_to_attribute = [('el - la', 1.0, el_minus_la)]
        elif continuation == ' la':
            quantity_to_attribute = [('la - el', 1.0, la_minus_el)]
        elif continuation == ' los':
            quantity_to_attribute = [('los - las', 1.0, los_minus_las)]
        elif continuation == ' las':
            quantity_to_attribute = [('las - los', 1.0, las_minus_los)]
        else:
            raise ValueError(f"Got bad continuation: '{continuation}'")

        with torch.inference_mode():
            logits = model(sentence)
            print_topk(model,logits)

        graph = attribute(sentence, model, batch_size=128, max_feature_nodes=7500, 
                        offload=None, verbose=True)

        names.append(name)

        pt_output_path = Path(f'attribution_graphs/{model_name}')
        pt_output_path.mkdir(exist_ok=True, parents=True)
        pt_output_path = pt_output_path / f'{name}.pt'
        graph.to_pt(pt_output_path)

        slug = f"{model_name}-{name}"

        json_output_path = './graph_files'
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

        del graph

        # graph = attribute(sentence, model, batch_size=128, max_feature_nodes=7500, 
        #                 offload=None, verbose=True, quantity_to_attribute=quantity_to_attribute)

        # pt_output_path = Path(f'attribution_graphs_diff/{model_short_name}')
        # pt_output_path.mkdir(exist_ok=True, parents=True)
        # pt_output_path = pt_output_path / f'{name}.pt'
        # graph.to_pt(pt_output_path)

        # slug = f"{model_short_name}-{name}"

        # json_output_path = './graph_files_diff'
        # create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

    df['filename'] = names
    Path('results/attribution_metadata').mkdir(exist_ok=True)
    df.to_csv(f'results/attribution_metadata/{model_name}.csv')
    del model
    torch.cuda.empty_cache()
# %%
