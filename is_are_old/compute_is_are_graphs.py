#%%
from pathlib import Path
from typing import List
from collections import namedtuple

import torch
import pandas as pd
import numpy as np

from circuit_tracer.attribution.attribute import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.upload_graph_to_s3 import upload_graph_to_s3


ATTRIB_DIFF = True
Example = namedtuple("Example", ["sentence", "continuation", "name"])

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
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

# Load the dataset
df = pd.read_csv('data/animals_dataset.csv')

# Sample 80 examples for each answer type (without replacement - default behavior)
is_samples = df[df['answer'] == 'is'].sample(n=80, random_state=42, replace=False)
are_samples = df[df['answer'] == 'are'].sample(n=80, random_state=42, replace=False)

df = pd.concat([is_samples, are_samples], ignore_index=True)
df['name'] = [f"{animal}-{original}-{subtracted}"
                for animal, original, subtracted in zip(df['animal'], df['original'], df['subtracted'])]

output_path = 'data/animals_dataset_downsampled.csv'
df.to_csv(output_path, index=False)

for model_name, transcoders in models_and_transcoders.items():
    model_short_name = model_name.split('/')[-1]
    print(model_short_name)
    model = ReplacementModel.from_pretrained(model_name, 
                                            transcoders, 
                                            lazy_encoder=True, 
                                            dtype=torch.bfloat16)

    is_token_id = model.tokenizer(" is").input_ids[0]
    are_token_id = model.tokenizer(" are").input_ids[0]

    print(model.W_E.size())
    is_vector = model.W_E[is_token_id]
    are_vector = model.W_E[are_token_id]

    # examples = [Example(f"{model.tokenizer.eos_token}{prompt}", f' {article}', f'{article}-{profession}') 
    #             for prompt, article, profession in zip(df_ex['Prompt'], df_ex['Article'], df_ex['Profession'])]

    for _, row in df.iterrows():
        sentence: str = row["prompt"].strip()

        instruction: str = f"Repeat this sentence and complete it. {sentence}"

        messages = [
            {
                "role": "user",
                "content": instruction
            }
        ]

        # Convert messages to Qwen3 chat format using tokenizer
        formatted_input = model.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Prefilled response (what the model should start generating after)
        prefill: str = f"<think>\n\n</think> {sentence}"

        # Combine the formatted input with prefilled response
        # The model will continue generating after the prefilled content
        prompt_base = formatted_input + prefill

        # input_ids = model.tokenizer(prompt_base).input_ids
        # tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        # print(tokens)
        # with torch.inference_mode():
        #     logits = model(prompt_base)
        #     print_topk(model,logits)

        if ATTRIB_DIFF:
            vector_to_attribute = is_vector - are_vector if row["answer"] == "is" else are_vector - is_vector
            str_to_attribute = 'is - are' if row['answer'] == 'is' else 'are - is'
            quantity_to_attribute = [(str_to_attribute, 1.0, vector_to_attribute)]
        else:
            quantity_to_attribute = None

        graph = attribute(prompt_base, model, quantity_to_attribute=quantity_to_attribute,batch_size=128, max_feature_nodes=7500, 
                        offload=None, verbose=True)
        name = f"{row['animal']}-{row['original']}-{row['subtracted']}"

        pt_output_path = Path(f'graphs_diff/{model_short_name}') if ATTRIB_DIFF else Path(f'graphs/{model_short_name}')
        pt_output_path.mkdir(exist_ok=True, parents=True)
        pt_output_path = pt_output_path / f'{name}.pt'
        graph.to_pt(pt_output_path)

        slug = f"{model_short_name}-{name}"

        json_output_path = Path(f'graph_files_diff/{model_short_name}') if ATTRIB_DIFF else Path(f'graph_files/{model_short_name}')
        json_output_path.mkdir(exist_ok=True, parents=True)
        json_output_path = json_output_path / f'{name}.pt'
        create_graph_files(graph, slug, json_output_path, node_threshold=0.8, edge_threshold=0.95)

    del model
    torch.cuda.empty_cache()
# %%
