#%%
from pathlib import Path
from typing import List
from collections import namedtuple

import torch
import pandas as pd
import numpy as np

from circuit_tracer import attribute, ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files


ATTRIB_DIFF = False
Example = namedtuple("Example", ["sentence", "continuation", "name"])

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

models_and_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
batch_size = []


def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

# Set seed for reproducibility
np.random.seed(42)


for model_name, transcoders in list(models_and_transcoders.items())[-1:]:
    print(model_name)
    model = ReplacementModel.from_pretrained('Qwen/' + model_name, 
                                            transcoders, 
                                            lazy_encoder=False, 
                                            dtype=torch.bfloat16)

    # Load the dataset
    df = pd.read_csv(f'results/behavioral/{model_name}.csv')

    # Sample 80 examples for each answer type (without replacement - default behavior)
    is_samples = df[df['gold_verb'] == 'is'].sample(n=80, random_state=42, replace=False)
    are_samples = df[df['gold_verb'] == 'are'].sample(n=80, random_state=42, replace=False)

    downsampled_df = pd.concat([is_samples, are_samples], ignore_index=True)
    downsampled_df['filename'] = [f"{animal}-{original}-{subtracted}"
                    for animal, original, subtracted in zip(downsampled_df['animal'], 
                                                            downsampled_df['original_number'], 
                                                            downsampled_df['subtracted_number'])]

    is_token_id = model.tokenizer(" is").input_ids[0]
    are_token_id = model.tokenizer(" are").input_ids[0]

    is_vector = model.W_E[is_token_id]
    are_vector = model.W_E[are_token_id]

    for _, row in downsampled_df.iterrows():
        prompt_base = row['prompt_base']

        if ATTRIB_DIFF:
            vector_to_attribute = is_vector - are_vector if row["gold_verb"] == "is" else are_vector - is_vector
            str_to_attribute = 'is - are' if row['gold_verb'] == 'is' else 'are - is'
            quantity_to_attribute = [(str_to_attribute, 1.0, vector_to_attribute)]
        else:
            quantity_to_attribute = None

        graph = attribute(prompt_base, model, quantity_to_attribute=quantity_to_attribute,batch_size=128, max_feature_nodes=7500, 
                        offload=None, verbose=True)
        name = f"{row['animal']}-{row['original_number']}-{row['subtracted_number']}"

        pt_output_path = Path(f'attribution_graphs_diff/{model_name}') if ATTRIB_DIFF else Path(f'attribution_graphs/{model_name}')
        pt_output_path.mkdir(exist_ok=True, parents=True)
        pt_output_path = pt_output_path / f'{name}.pt'
        graph.to_pt(pt_output_path)

        slug = f"{model_name}-{name}"

        json_output_path = Path(f'graph_files_diff/{model_name}') if ATTRIB_DIFF else Path(f'graph_files/{model_name}')
        json_output_path.mkdir(exist_ok=True, parents=True)
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

    Path('results/attribution_metadata').mkdir(exist_ok=True)
    downsampled_df.to_csv(f'results/attribution_metadata/{model_name}.csv')
    del model
    torch.cuda.empty_cache()
# %%
