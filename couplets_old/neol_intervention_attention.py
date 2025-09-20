#%%
from functools import partial
from collections import defaultdict
from pathlib import Path
import torch
import pandas as pd
from circuit_tracer import Graph, ReplacementModel

from load_feature_from_binary import get_features_top_acts_from_list
#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}

def is_near_EOL_feature(feature_info):
    near_EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        near_EOL_tokens = tokens[top_index + 2: top_index + 5]
        if any('⏎' in tok for tok in near_EOL_tokens):
            near_EOL_count += 1
    return near_EOL_count >= 7

def get_features_with_cache(features: list[tuple[int,int]], cache: dict, model_name: str, verbose=False):
    features_to_get = [feature for feature in features if feature not in cache]
    if features_to_get:
        new_features = get_features_top_acts_from_list(model_name, features_to_get, verbose=verbose)
        cache.update(new_features)
    return {feature: cache[feature] for feature in features if cache[feature] is not None}
#%%
for model_name in models:
    feature_info_cache = {}
    whole_model_name = f"Qwen/{model_name}"
    transcoders_name = models_and_transcoders[whole_model_name]
    model = ReplacementModel.from_pretrained(whole_model_name, 
                                             transcoders_name,
                                             dtype=torch.bfloat16, 
                                             lazy_encoder=('8B' in model_name or '14B' in model_name))

    graph_dir = Path(f'attribution_graphs/{model_name}')
    metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)

    pre_attns = []
    post_attns = []
    original_inputs = []

    feature_set = set()
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        original_input = model.tokenizer.decode(graph.input_tokens)
        original_inputs.append(original_input)
        last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
        last_word_features_unique = set((layer, feature) for layer, _, feature in last_word_features.tolist())
        feature_set |= last_word_features_unique
        
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)

    for idx, row in metadata.head(10).iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)
        last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
        last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
        last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
        neol_features = {k:v for k,v in last_word_feature_infos.items() if is_near_EOL_feature(v)}
    
        selected_features = graph.active_features[graph.selected_features]
        selected_last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_last_word_features_unique = set((layer, feature) for layer, _, feature in selected_last_word_features.tolist())
        selected_feature_count = sum(eol_feature in selected_last_word_features_unique for eol_feature in neol_features.keys())

        acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                     graph.activation_values, 
                                     size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))
    
        # take some arbitrary substring
        substring = model.tokenizer.decode(graph.input_tokens[:input_tokens.index(model.tokenizer.eos_token) + 14])

        # normal generation is just what we observe
        _, cache = model.run_with_cache(substring)
        pre_attns.append(torch.stack([cache[f'blocks.{n}.attn.hook_pattern'][0, :, -1, last_word - 2] for n in range(model.cfg.n_layers)]))
    
        # continue generation
        continue_interventions = [(layer, slice(-1, None), feature, -6 * acts[layer, graph.n_pos - 1, feature]) for layer, feature in neol_features.keys()]
        cache, fwd, bwd = model.get_caching_hooks()
        with model.hooks(fwd):
            continued_generation, _, = model.feature_intervention(substring, continue_interventions, return_activations=False)

        post_attns.append(torch.stack([cache[f'blocks.{n}.attn.hook_pattern'][0, :, -1, last_word - 2] for n in range(model.cfg.n_layers)]))

    pre_stack = torch.stack(pre_attns)
    post_stack = torch.stack(post_attns)

    layers_to_heads = defaultdict(list)
    
    values, indices = (pre_stack - post_stack).mean(0).view(-1).topk(5)
    for idx, value in zip(indices, values):
        layer, head = torch.unravel_index(idx, pre_stack.size()[1:])
        layers_to_heads[layer.item()].append(head.item())

    print(layers_to_heads)

    substrings, original_generations, ablated_generations = [], [], []
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)
        last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
        last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
        last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
        neol_features = {k:v for k,v in last_word_feature_infos.items() if is_near_EOL_feature(v)}
      
        selected_features = graph.active_features[graph.selected_features]
        selected_last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_last_word_features_unique = set((layer, feature) for layer, _, feature in selected_last_word_features.tolist())
        selected_feature_count = sum(eol_feature in selected_last_word_features_unique for eol_feature in neol_features.keys())
      
        # take some arbitrary substring
        toks = graph.input_tokens[:input_tokens.index(model.tokenizer.eos_token) + 14]
        substring = model.tokenizer.decode(toks)
        substrings.append(substring)

        original_generation = model.generate(substring, do_sample=False, max_new_tokens=20)
        original_generations.append(original_generation)

        def attention_intervention(acts, hook, heads, last_word):
            for head in heads:
                acts[0, head, -1, 0] += acts[0, head, -1, last_word - 2]
                acts[0, head, -1, last_word - 2] = 0
            return acts


        hooks = [(f'blocks.{layer}.attn.hook_pattern', partial(attention_intervention, heads=heads, last_word=last_word)) for layer, heads in layers_to_heads.items()]
        with model.hooks(hooks):
            ablated_generation = model.generate(substring, do_sample=False, max_new_tokens=20)
        ablated_generations.append(ablated_generation)

    metadata['substring'] = substrings
    metadata['original_generation'] = original_generations
    metadata['ablated_generation'] = ablated_generations
    metadata['original_input'] = original_inputs
    
    # Ensure results directory exists
    results_dir = Path('results/attention_intervention')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metadata.to_csv(results_dir / f'{model_name}.csv')
#%%
# model_name = models[-1]
# feature_info_cache = {}
# whole_model_name = f"Qwen/{model_name}"
# transcoders_name = models_and_transcoders[whole_model_name]
# model = ReplacementModel.from_pretrained(whole_model_name, 
#                                             transcoders_name,
#                                             dtype=torch.bfloat16, 
#                                             lazy_encoder=('8B' in model_name or '14B' in model_name))

# graph_dir = Path(f'attribution_graphs/{model_name}')
# metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)

# substrings = []
# stopped_generations = []
# original_inputs = []
# original_generations = []
# continued_generations = []
# n_features = []
# n_selected_features = []
# feature_set = set()
# for idx, row in metadata.iterrows():
#     second_last_word = row['second_last_word']
#     graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
#     last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
#     last_word_features_unique = set((layer, feature) for layer, _, feature in last_word_features.tolist())
#     feature_set |= last_word_features_unique
    
# get_features_with_cache(list(feature_set), feature_info_cache, model_name)
# #%%
# metadata = metadata.head(10)
# all_pre_attns = []
# all_post_attns = []
# pre_attns = []
# post_attns = []
# for idx, row in metadata.iterrows():
#     second_last_word = row['second_last_word']
#     graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
#     input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
#     last_word = input_tokens.index(model.tokenizer.eos_token)
#     last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
#     last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
#     last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
#     neol_features = {k:v for k,v in last_word_feature_infos.items() if is_near_EOL_feature(v)}
#     n_features.append(len(neol_features))

#     selected_features = graph.active_features[graph.selected_features]
#     selected_last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
#     selected_last_word_features_unique = set((layer, feature) for layer, _, feature in selected_last_word_features.tolist())
#     selected_feature_count = sum(eol_feature in selected_last_word_features_unique for eol_feature in neol_features.keys())
#     n_selected_features.append(selected_feature_count)

#     # take some arbitrary substring
#     substring = model.tokenizer.decode(graph.input_tokens[:input_tokens.index(model.tokenizer.eos_token) + 12])

#     # normal generation is just what we observe
#     print(substring)
#     # stop generation
#     acts = torch.sparse_coo_tensor(graph.active_features.t(), 
#                                     graph.activation_values, 
#                                     size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))

#     # normal generation
#     _, cache = model.run_with_cache(substring)
#     pre_attns.append(torch.stack([cache[f'blocks.{n}.attn.hook_pattern'][0, :, -1, last_word - 2] for n in range(model.cfg.n_layers)]))
#     all_pre_attns.append(torch.stack([cache[f'blocks.{n}.attn.hook_pattern'] for n in range(model.cfg.n_layers)]))

#     # continue generation
#     continue_interventions = [(layer, slice(-1, None), feature, -6 * acts[layer, graph.n_pos - 1, feature]) for layer, feature in neol_features.keys()]
#     cache, fwd, bwd = model.get_caching_hooks()
#     with model.hooks(fwd):
#         continued_generation, _, = model.feature_intervention(substring, continue_interventions, return_activations=False)

#     post_attns.append(torch.stack([cache[f'blocks.{n}.attn.hook_pattern'][0, :, -1, last_word - 2] for n in range(model.cfg.n_layers)]))
#     all_post_attns.append(torch.stack([cache[f'blocks.{n}.attn.hook_pattern'] for n in range(model.cfg.n_layers)]))
# # %%
# pre_stack = torch.stack(pre_attns)
# post_stack = torch.stack(post_attns)
# #%%
# values, indices = (pre_stack - post_stack).mean(0).view(-1).topk(5)
# for head, value in zip(indices, values):
#     print(torch.unravel_index(head, pre_stack.size()[1:]), value)
# # %%
# def attention_intervention(acts, hook, initial_length, last_word):
#     acts[0, 12, initial_length:, 0] += acts[0, 12, initial_length:, last_word - 2]
#     acts[0, 12, initial_length:, last_word - 2] = 0
    
#     #acts[0, 14, initial_length:, 0] += acts[0, 14, initial_length:, last_word - 2]
#     #acts[0, 14, initial_length:, last_word - 2] = 0
#     return acts

# #%%
# from functools import partial
# for idx, row in metadata.iterrows():
#     second_last_word = row['second_last_word']
#     graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
#     input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
#     last_word = input_tokens.index(model.tokenizer.eos_token)
#     last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
#     last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
#     last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
#     neol_features = {k:v for k,v in last_word_feature_infos.items() if is_near_EOL_feature(v)}
#     n_features.append(len(neol_features))

#     selected_features = graph.active_features[graph.selected_features]
#     selected_last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
#     selected_last_word_features_unique = set((layer, feature) for layer, _, feature in selected_last_word_features.tolist())
#     selected_feature_count = sum(eol_feature in selected_last_word_features_unique for eol_feature in neol_features.keys())
#     n_selected_features.append(selected_feature_count)

#     # take some arbitrary substring
#     toks = graph.input_tokens[:input_tokens.index(model.tokenizer.eos_token) + 14]
#     substring = model.tokenizer.decode(toks)
#     def attention_intervention(acts, hook, last_word):
#         print(acts[0, 12, -1, last_word - 2])
#         print(acts[0, 14, -1, last_word - 2])
#         acts[0, 12, -1, 0] += acts[0, 12, -1, last_word - 2]
#         acts[0, 12, -1, last_word - 2] = 0
        
#         acts[0, 14, -1, 0] += acts[0, 14, -1, last_word - 2]
#         acts[0, 14, -1, last_word - 2] = 0
#         return acts

#     hooks = [('blocks.30.attn.hook_pattern', partial(attention_intervention, last_word=last_word))]
#     with model.hooks(hooks):
#         generation = model.generate(substring, do_sample=False)
    
#     print(generation)
# # %%
