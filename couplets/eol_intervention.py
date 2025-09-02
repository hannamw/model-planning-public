#%%
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


def is_EOL_feature(feature_info: dict):
    EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
            EOL_count += 1
    return EOL_count >= 7

def get_features_with_cache(features: list[tuple[int,int]], cache: dict, model_name: str):
    features_to_get = [feature for feature in features if feature not in cache]
    new_features = get_features_top_acts_from_list(model_name, features_to_get)
    cache.update(new_features)
    return {feature: cache[feature] for feature in features if feature in cache}

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

    substrings = []
    stopped_generations = []
    whole_couplets = []
    original_generations = []
    continued_generations = []
    n_features = []
    n_selected_features = []
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)
        last_word_features = graph.active_features[graph.active_features[:, 1] == last_word - 2]
        last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
        last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
        eol_features = {k:v for k,v in last_word_feature_infos.items() if is_EOL_feature(v)}
        n_features.append(len(eol_features))

        selected_features = graph.active_features[graph.selected_features]
        selected_last_word_features = selected_features[selected_features[:, 1] == last_word - 2]
        selected_last_word_features_unique = set((layer, feature) for layer, _, feature in selected_last_word_features.tolist())
        selected_feature_count = sum(eol_feature in selected_last_word_features_unique for eol_feature in eol_features.keys())
        n_selected_features.append(selected_feature_count)

        # take some arbitrary substring
        substring = model.tokenizer.decode(graph.input_tokens[:last_word + 12])

        # normal generation is just what we observe

        # stop generation
        acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                        graph.activation_values, 
                                        size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))
        stop_interventions = [(layer, -1, feature, 2 * acts[layer, last_word - 2, feature]) 
                                for layer, feature in eol_features.keys()]

        stopped_generation, _, _ = model.feature_intervention_generate(substring, stop_interventions, do_sample=False)

        # take the whole string
        whole_couplet = model.tokenizer.decode(graph.input_tokens) + ' ' + second_last_word

        # normal generation
        original_generation = model.generate(whole_couplet, do_sample=False)

        # continue generation
        continue_interventions = [(layer, slice(-1, None), feature, -2 * acts[layer, last_word - 2, feature]) for layer, feature in eol_features.keys()]
        continued_generation, _, _ = model.feature_intervention_generate(whole_couplet, continue_interventions, do_sample=False)
        substrings.append(substring)
        stopped_generations.append(stopped_generation)
        whole_couplets.append(whole_couplet)
        original_generations.append(original_generation)
        continued_generations.append(continued_generation)


    metadata['n_features'] = n_features
    metadata['n_selected_features'] = n_selected_features
    metadata['substring'] = substrings
    metadata['stopped_generation'] = stopped_generations
    metadata['whole_couplet'] = whole_couplets
    metadata['original_generation'] = original_generations
    metadata['continued_generation'] = continued_generations
    metadata.to_csv(f'results/eol_intervention/{model_name}.csv')