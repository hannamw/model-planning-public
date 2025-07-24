#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import argparse

import torch
import pandas as pd
from tqdm import tqdm

from circuit_tracer.attribution import attribute
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils.create_graph_files import create_graph_files
from circuit_tracer.frontend.upload_graph_to_s3 import upload_graph_to_s3

Example = namedtuple("Example", ["sentence", "coninuation", "name"])

def print_topk(model, logits:torch.Tensor, k=5):
    probs = torch.softmax(logits.squeeze()[-1], dim=-1)
    topk = torch.topk(probs, k)
    for i in range(k):
        print(model.tokenizer.decode([topk.indices[i]]), ':', topk.values[i].item())

MODEL_CONFIGS = {
    'qwen3-0.6b': ('Qwen/Qwen3-0.6B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-0.6b-relu-lowl0.yaml'),
    'qwen3-1.7b': ('Qwen/Qwen3-1.7B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-1.7b-relu-lowl0.yaml'),
    'qwen3-4b': ('Qwen/Qwen3-4B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-4b-relu.yaml'),
    'qwen3-8b': ('Qwen/Qwen3-8B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-8b-relu.yaml'),
    'qwen3-14b': ('Qwen/Qwen3-14B', '../circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'),
}

def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

parser = argparse.ArgumentParser(description='Run attribution for selected Qwen models')
parser.add_argument('--models', nargs='+', choices=list(MODEL_CONFIGS.keys()),
                    help='List of models to run attribution for')
parser.add_argument('--all', action='store_true', help='Run attribution for all models')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing attribution graph files')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--transcoders-offload', type=str, default='disk')
args = parser.parse_args()

if not args.models and not args.all:
    parser.error("Either specify --models or use --all")

selected_models = list(MODEL_CONFIGS.keys()) if args.all else args.models

df = pd.read_csv('data/filtered_multihop_dataset.csv')

for model_key in selected_models:
    model_name, model_config = MODEL_CONFIGS[model_key]
    model_short_name = Path(model_config).stem
    print(model_short_name)
    model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload=args.transcoders_offload, 
                                            dtype=torch.bfloat16)

    base_prompt = prompt_template = ["/no_think Answer the following question in one word. Q: {question}", 
                        "<think>\n\n</think>\n\nA:"]
    examples = [Example(chattify(base_prompt, model.tokenizer).format(question=prompt), f' {answer.strip()}', f'{prompt_type}-{answer}') 
                for prompt, answer, prompt_type in zip(df['question'], df['answer'], df['prompt_type'])]

    for sentence, continuation, name in tqdm(examples, desc=f"Processing {model_key}"):
        pt_output_path = Path(f'attribution_graphs/{model_short_name}')
        pt_output_path.mkdir(exist_ok=True, parents=True)
        pt_output_path = pt_output_path / f'{name}.pt'
        
        if pt_output_path.exists() and not args.overwrite:
            print(f"Skipping {name} - file already exists")
            continue

        input_ids = model.tokenizer(sentence).input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        with torch.inference_mode():
            logits = model(sentence)
            print_topk(model,logits)

        graph = attribute(sentence, model, batch_size=args.batch_size, max_feature_nodes=7500, 
                        offload=None, verbose=False)
            
        graph.to_pt(pt_output_path)

        slug = f"{model_short_name}-{name}"

        #upload_graph_to_s3(output_path, slug, node_threshold=0.8, edge_threshold=0.95)
        #print(f"Graph now available at http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug={slug}")

    del model
    torch.cuda.empty_cache()
# %%
