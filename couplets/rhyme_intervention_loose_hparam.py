#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from pathlib import Path
import re
from collections import Counter
from functools import partial, lru_cache

import pandas as pd
import torch

from circuit_tracer import Graph, ReplacementModel
from load_feature_from_binary import get_features_top_acts_from_list
from rhyme_eval_utils import eval_df

def term_in_logits(term:str, top: list[str], bottom: list[str], use_bottom=True, substring_ok=True, k=5):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    term = term.strip().lower()
    if substring_ok:
        len1 = 0
        for logit in logits:
            if logit == '':
                continue
            if term.startswith(logit):
                if len(logit) == 1:
                    len1 += 1
                    if len1 >= 2:
                        return True
                else:
                    return True
        return False
    else:
        return any(term in logit for logit in logits)

def is_EOL_feature(feature_info):
    EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
            EOL_count += 1
    return EOL_count >= 3 or any('⏎' in logit for logit in feature_info['top_logits'])

def is_near_EOL_feature(feature_info):
    near_EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        near_EOL_tokens = tokens[top_index + 2: top_index + 5]
        if any('⏎' in tok for tok in near_EOL_tokens):
            near_EOL_count += 1
    return near_EOL_count >= 3

def is_near_word_feature(feature_info, word, exclude=''):
    word_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        s = max(top_index-2, 0)
        e = min(top_index + 2, len(tokens) - 1)
        if word in ''.join(tokens[s:e]).lower():
            word_count += 1
    return word_count >= 3

def _is_not_other_feature(layer, feature_idx, word, n_layers, feature_cache, exclude=None):
    feature_info = feature_cache[(layer, feature_idx)]
    return (layer > n_layers / 2) and not (is_EOL_feature(feature_info) or is_near_EOL_feature(feature_info) 
                                or is_near_word_feature(feature_info, word) or feature_info['frequency'] > 0.05)

def _is_rhyme_feature(layer, feature_idx, word, feature_cache, exclude=None):
    feature_info = feature_cache[(layer, feature_idx)]
    first_chars, last_chars = Counter(), Counter()
    token_counts = Counter()
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        token = tokens[top_index]
        token = re.sub(r'^[^\w]+|[^\w]+$', '', token.strip()).lower()
        if len(token) >  4:
            return False
        if not token:
            continue
        first_chars[token[0]] += 1
        last_chars[token[-1]] += 1
        token_counts[token] += 1
    if len(first_chars) == 0:
        return False
    most_common_first_char, first_count = first_chars.most_common(1)[0]
    most_common_last_char, last_count = last_chars.most_common(1)[0]
    most_common_token, token_count = token_counts.most_common(1)[0]
    return (token_count <= 5) and (exclude is None or exclude not in most_common_token) and ((most_common_first_char in 'aeiou' and first_count >=7) or last_count >=7)

def get_features_with_cache(features: list[tuple[int,int]], cache: dict, model_name: str, verbose=False):
    features_to_get = [feature for feature in features if feature not in cache]
    if features_to_get:
        new_features = get_features_top_acts_from_list(model_name, features_to_get, verbose=verbose)
        cache.update(new_features)
    return {feature: cache[feature] for feature in features if cache[feature] is not None}

models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
#%%
model_name = models[-2]
feature_info_cache = {}

is_rhyme_word_feature = lru_cache(maxsize=None)(
    partial(_is_rhyme_feature, feature_cache=feature_info_cache)
)

is_not_other_feature = lru_cache(maxsize=None)(
    partial(_is_not_other_feature, feature_cache=feature_info_cache)
)


whole_model_name = f"Qwen/{model_name}"
transcoders_name = models_and_transcoders[whole_model_name]
model = ReplacementModel.from_pretrained(whole_model_name, 
                                            transcoders_name,
                                            dtype=torch.bfloat16, 
                                            lazy_encoder=('8B' in model_name or '14B' in model_name))

graph_dir = Path(f'attribution_graphs/{model_name}')
metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)

feature_set = set()

id_to_features_acts = {}
for idx, row in metadata.iterrows():
    second_last_word = row['second_last_word']
    graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")

    input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
    last_word = input_tokens.index(model.tokenizer.eos_token)

    selected_features = graph.active_features[graph.selected_features]
    last_word_features = selected_features[selected_features[:, 1] == last_word - 2]
    feature_set.update((layer, feature) for layer, _, feature in last_word_features.tolist())

# get all features at once
get_features_with_cache(list(feature_set), feature_info_cache, model_name)

for idx, row in metadata.iterrows():
    second_last_word = row['second_last_word']
    graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
    input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
    last_word = input_tokens.index(model.tokenizer.eos_token)


    acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                    graph.activation_values, 
                                    size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))

    selected_features = graph.active_features[graph.selected_features]
    last_word_features = selected_features[selected_features[:, 1] == last_word - 2]
    last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
    last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
    #rhyme_features = {k:v for k,v in last_word_feature_infos.items() if is_rhyme_word_feature(*k, second_last_word)}
    rhyme_features = {k:v for k,v in last_word_feature_infos.items() if is_not_other_feature(*k, second_last_word, model.cfg.n_layers)}
    
    id_to_features_acts[f"{idx}-{second_last_word}"] = [(layer, feat, acts[layer, last_word - 2, feat]) 
                                                        for layer, feat in rhyme_features.keys()]

metadata['feature_count'] = [len(id_to_features_acts[f"{idx}-{second_last_word}"] )
                            for idx, second_last_word in zip(metadata.index, metadata['second_last_word'])]
#%%
metadata = metadata.head(20)
for downweight_coeff, upweight_coeff in [(-3, 5), (-3, 6), (-3, 7), (-4, 6), (-4, 7)]:
    # Initialize columns for intervention results
    metadata['found_valid_row'] = False
    metadata['chosen_id'] = ''
    metadata['chosen_index'] = 0
    metadata['chosen_rhyme_group'] = ''
    metadata['chosen_feature_count'] = 0
    metadata['original_generation'] = ''
    metadata['intervention_generation'] = ''

    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        rhyme_group = row['rhyme_group']
        id = f"{idx}-{second_last_word}"
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)

        # Generate original text
        unthink_idx = input_tokens.index('</think>')
        input_text = model.tokenizer.decode(graph.input_tokens[:unthink_idx + 2])
        original_generation = model.generate(input_text, do_sample=False)
        metadata.at[idx, 'original_generation'] = original_generation

        # select another word to replace it with
        # different word, does have features, different rhyme group?
        valid_rows = metadata[(metadata['feature_count'] > 0) & (metadata['rhyme_group'] != rhyme_group)]

        if not valid_rows.empty:
            chosen_row_raw = valid_rows.sample(1)
            chosen_row = chosen_row_raw.iloc[0]
            chosen_index = chosen_row_raw.index[0]
            chosen_rhyme_group = chosen_row['rhyme_group']
            chosen_second_last_word = chosen_row['second_last_word']
            chosen_id = f'{chosen_index}-{chosen_second_last_word}'
            chosen_feature_count = chosen_row['feature_count']
            
            # Store successful intervention metadata
            metadata.at[idx, 'found_valid_row'] = True
            metadata.at[idx, 'chosen_id'] = chosen_id
            metadata.at[idx, 'chosen_index'] = chosen_index
            metadata.at[idx, 'chosen_rhyme_group'] = chosen_rhyme_group
            metadata.at[idx, 'chosen_feature_count'] = chosen_feature_count
            
            acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                            graph.activation_values, 
                                            size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))
            
            downweight_interventions = [(layer, last_word - 2, feature, downweight_coeff * act) 
                                    for layer, feature, act in id_to_features_acts[id]]
            new_word_interventions = [(layer, last_word - 2, feature, upweight_coeff * act) 
                                    for layer, feature, act in id_to_features_acts[chosen_id]]

            new_generation, new_logits, _ = model.feature_intervention_generate(
                input_text, 
                interventions=downweight_interventions + new_word_interventions, 
                do_sample=False, 
                return_activations=False,
                max_new_tokens=20
            )
            
            # Store intervention generation
            metadata.at[idx, 'intervention_generation'] = new_generation
            
        else:
            print(f"couldn't find valid row for {id}; using dummy data")
            # Store dummy data for failed cases
            metadata.at[idx, 'found_valid_row'] = False
            metadata.at[idx, 'chosen_id'] = 'DUMMY_NO_VALID_ROW'
            metadata.at[idx, 'chosen_index'] = -999
            metadata.at[idx, 'chosen_rhyme_group'] = 'DUMMY_RHYME_GROUP'
            metadata.at[idx, 'chosen_feature_count'] = -999
            metadata.at[idx, 'intervention_generation'] = 'DUMMY_INTERVENTION_GENERATION'
    print(downweight_coeff, upweight_coeff)
    eval_df(metadata)

# %%
