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


def is_EOL_feature(feature_info: dict):
    EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
            EOL_count += 1
    return EOL_count >= 7

def get_features_with_cache(features: list[tuple[int,int]], cache: dict, model_name: str, verbose=False):
    features_to_get = [feature for feature in features if feature not in cache]
    if features_to_get:
        new_features = get_features_top_acts_from_list(model_name, features_to_get, verbose=verbose)
        cache.update(new_features)
    return {feature: cache[feature] for feature in features if cache[feature] is not None}

for model_name in models:
    feature_info_cache = {}
    whole_model_name = f"Qwen/{model_name}"
    transcoders_name = models_and_transcoders[whole_model_name]
    model = ReplacementModel.from_pretrained(whole_model_name, 
                                             transcoders_name,
                                             dtype=torch.bfloat16)

    graph_dir = Path(f'attribution_graphs/{model_name}')
    metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)

    substrings = []
    stopped_generations = []
    #new_final_logits = []
    n_features = []
    feature_set = set()
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)
        last_word_features = graph.active_features[graph.active_features[:, 1] == last_word - 2]
        last_word_features_unique = set((layer, feature) for layer, _, feature in last_word_features.tolist())
        feature_set |= last_word_features_unique
        
    get_features_with_cache(list(feature_set), feature_info_cache, model_name)

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

        # take some arbitrary substring
        end_index = input_tokens.index('</think>')
        substring = model.tokenizer.decode(graph.input_tokens[:end_index + 2])
        substrings.append(substring)

        # stop generation
        acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                        graph.activation_values, 
                                        size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))
        stop_interventions = [(layer, last_word - 2, feature, -4 * acts[layer, last_word - 2, feature]) 
                                for layer, feature in eol_features.keys()]

        stopped_generation, _, _ = model.feature_intervention_generate(substring, stop_interventions, do_sample=False, max_new_tokens=20)

        final_logits, _ = model.feature_intervention(model.tokenizer.decode(graph.input_tokens), stop_interventions)
        #top_logit = torch.topk(final_logits.squeeze(0)[-1]).indices[0]
        stopped_generations.append(stopped_generation)
        #new_final_logits.append(top_logit)

    metadata['n_features'] = n_features
    metadata['substring'] = substrings
    metadata['stopped_generation'] = stopped_generations
    
    # Ensure results directory exists
    results_dir = Path('results/eol_intervention_rhyme')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metadata.to_csv(results_dir / f'{model_name}.csv')
    del model
    torch.cuda.empty_cache()