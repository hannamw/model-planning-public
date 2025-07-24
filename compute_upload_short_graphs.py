#%%
from pathlib import Path
from typing import List
from collections import namedtuple
import torch

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

model_names_and_configs = [
    #('Qwen/Qwen3-0.6B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-0.6b-relu-lowl0.yaml'),
    ('Qwen/Qwen3-1.7B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-1.7b-relu-lowl0.yaml'),
    #('Qwen/Qwen3-4B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-4b-relu.yaml'),
    #('Qwen/Qwen3-8B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-8b-relu.yaml'),
    #('Qwen/Qwen3-14B', 'circuit-tracer-dev/circuit_tracer/configs/qwen3-14b-relu-lowl0.yaml'),
    ]

# prompts = ["/no_think Repeat the following sentence and complete it. At first there were 4 cats. Then, 3 went away. Now, there", 
# "At first there were 4 cats. Then, 3 went away. Now, there"]
# next_token = "is"
# prompt_name = "is-4-3-cats"

# prompts = ["Someone who studies living organisms is a biologist. Someone who mixes and serves drinks is"]
# next_token = "a"
# prompt_name = "a-bartender"

def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

base_examples = [
    # Example(["Someone who dispenses medications is a pharmacist. Someone who studies stars and planets is"],
    # 'an', 'an-astronomer'),
    # Example(["Someone who dispenses medications is a pharmacist. Someone who presides over legal proceedings is"],
    # 'a', 'a-judge'),
    # Example(["Someone who leads orchestras or manages trains is a conductor. Someone who reviews and revises written content is"],
    # 'an', 'an-editor'),
    # Example(["Someone who leads orchestras or manages trains is a conductor. Someone who analyzes data and calculates probabilities is"],
    # 'a', 'a-statistician'),
    Example(['Someone who studies matter and energy is a physicist. Someone who conducts auctions is'], 'an', 'an-auctioneer')
]

for model_name, model_config in model_names_and_configs:
    model_short_name = Path(model_config).stem
    print(model_short_name)
    model = ReplacementModel.from_pretrained(model_name, 
                                            model_config, 
                                            transcoders_offload='disk', 
                                            dtype=torch.bfloat16)

    examples = [Example(chattify(x, model.tokenizer), y, z) for x,y,z in base_examples]

    for sentence, continuation, name in examples:
        input_ids = model.tokenizer(sentence).input_ids
        tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
        print(tokens)
        with torch.inference_mode():
            logits = model(sentence)
            print_topk(model,logits)

        graph = attribute(sentence, model, batch_size=128, max_feature_nodes=7500, 
                        offload=None, verbose=True)

        output_path = Path(f'attribution_graphs/{model_short_name}')
        output_path.mkdir(exist_ok=True, parents=True)
        output_path = output_path / f'{name}.pt'

        graph.to_pt(output_path)

        slug = f"{model_short_name}-{name}"

        upload_graph_to_s3(output_path, slug, node_threshold=0.8, edge_threshold=0.95)
        print(f"Graph now available at http://afp-circuit-tracing.s3-website-us-west-2.amazonaws.com/?slug={slug}")

    del model
    torch.cuda.empty_cache()

# %%
