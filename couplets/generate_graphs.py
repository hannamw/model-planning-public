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

model_names_and_configs = [
    ('Qwen/Qwen3-0.6B', 'mwhanna/qwen3-0.6b-transcoders-lowl0'),
    ('Qwen/Qwen3-1.7B', 'mwhanna/qwen3-1.7b-transcoders-lowl0'),
    ('Qwen/Qwen3-4B', 'mwhanna/qwen3-4b-transcoders'),
    ('Qwen/Qwen3-8B', 'mwhanna/qwen3-8b-transcoders'),
    ('Qwen/Qwen3-14B', 'mwhanna/qwen3-14b-transcoders-lowl0'),
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


for model_name, model_config in model_names_and_configs:
    model_short_name = model_name.split('/')[-1]
    print(model_short_name)
    model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            lazy_encoder=('14B' in model_short_name),
                                            dtype=torch.bfloat16)

    df = pd.read_csv(f'results/couplets-fixed/{model_short_name}.csv')
    raw_completion_without_last_word = [' '.join(comp.split(' ')[:-1]) for comp in df['raw_completion']]
    df['raw_completion_without_last'] = raw_completion_without_last_word
    df_filtered = df[df['rhyme_success']]
    df_ex = df_filtered.head(100)

    examples = [Example(chattify([f'/no_think Write only the next line of this rhyming couplet: {prompt}', f'{raw_completion}'], model.tokenizer), f' {last_word}', f'{idx}-{last_word}') 
                for prompt, raw_completion, last_word, idx in zip(df_ex['first_line'], df_ex['raw_completion_without_last'], df_ex['second_last_word'], df_ex.index)]

    for sentence, continuation, name in examples:
        input_ids = model.tokenizer(sentence).input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        print(continuation)

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

        json_output_path = './graph_files'
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

    df_ex.to_csv(f'results/attribution_metadata/{model_short_name}.csv')

    del model
    torch.cuda.empty_cache()

# %%
