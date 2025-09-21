#%%
from pathlib import Path
import torch
import pandas as pd
from circuit_tracer import Graph, ReplacementModel

from load_feature_from_binary import get_features_top_acts_from_list
#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]][-1:]

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
                                             lazy_encoder=False)

    graph_dir = Path(f'attribution_graphs/{model_name}')
    metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)

    substrings = []
    stopped_generations = []
    original_inputs = []
    original_generations = []
    continued_generations = []
    n_features = []
    n_selected_features = []
    feature_set = set()
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
        last_word_features_unique = set((layer, feature) for layer, _, feature in last_word_features.tolist())
        feature_set |= last_word_features_unique
        
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)

    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word_features = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1]
        last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
        last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
        neol_features = {k:v for k,v in last_word_feature_infos.items() if is_near_EOL_feature(v)}
        n_features.append(len(neol_features))

        selected_features = graph.active_features[graph.selected_features]
        selected_last_word_features = selected_features[selected_features[:, 1] == graph.n_pos - 1]
        selected_last_word_features_unique = set((layer, feature) for layer, _, feature in selected_last_word_features.tolist())
        selected_feature_count = sum(eol_feature in selected_last_word_features_unique for eol_feature in neol_features.keys())
        n_selected_features.append(selected_feature_count)

        # take some arbitrary substring
        substring = model.tokenizer.decode(graph.input_tokens[:input_tokens.index(model.tokenizer.eos_token) + 12])

        # normal generation is just what we observe

        # stop generation
        acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                        graph.activation_values, 
                                        size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))
        stop_interventions = [(layer, slice(-1, None), feature, 2 * acts[layer, graph.n_pos - 1, feature]) 
                                for layer, feature in neol_features.keys()]

        stopped_generation, _, _ = model.feature_intervention_generate(substring, stop_interventions, do_sample=False, return_activations=False)


        # take the whole string
        original_input = model.tokenizer.decode(graph.input_tokens)

        # normal generation
        original_generation = model.generate(original_input, do_sample=False)

        # continue generation
        continue_interventions = [(layer, slice(-1, None), feature, -2 * acts[layer, graph.n_pos - 1, feature]) for layer, feature in neol_features.keys()]
        continued_generation, _, _ = model.feature_intervention_generate(original_input, continue_interventions, do_sample=False, return_activations=False)

        substrings.append(substring)
        stopped_generations.append(stopped_generation)
        original_inputs.append(original_input)
        original_generations.append(original_generation)
        continued_generations.append(continued_generation)

    metadata['n_features'] = n_features
    metadata['n_selected_features'] = n_selected_features
    metadata['substring'] = substrings
    metadata['stopped_generation'] = stopped_generations
    metadata['original_input'] = original_inputs
    metadata['original_generation'] = original_generations
    metadata['continued_generation'] = continued_generations
    
    # Ensure results directory exists
    results_dir = Path('results/neol_intervention')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metadata.to_csv(results_dir / f'{model_name}.csv')
#%%