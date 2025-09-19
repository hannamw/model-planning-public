#%%
import re
from collections import defaultdict
from functools import partial, lru_cache
import random
from pathlib import Path

import torch
import pandas as pd
from datasets import load_dataset


from circuit_tracer import ReplacementModel, Graph
from load_feature_from_binary import get_features_top_acts_from_list
#%%
def _chattify(inputs: list[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

# Helper function to strip the model's internal "thinking" preamble before displaying.
def strip_think(text: str) -> str:
    return re.sub('</think>\n\n', '', text, flags=re.DOTALL)

def _get_topk(logits, tokenizer):
    logits = logits.squeeze(0)[-1]
    probs = torch.softmax(logits, -1)
    values, indices = torch.topk(probs, k=5)
    return [(tokenizer.decode(index), value.item()) for value, index in zip(values, indices)]

    
def _get_features_with_cache(features: list[tuple[int,int]], cache: dict, model_name: str, verbose=False):
    features_to_get = [feature for feature in features if feature not in cache]
    if features_to_get:
        new_features = get_features_top_acts_from_list(model_name, features_to_get, verbose=verbose)
        cache.update(new_features)
    return {feature: cache[feature] for feature in features if cache[feature] is not None}

# get the dataset to steer on
def term_in_logits(term:str, top: list[str], bottom: list[str], use_bottom=True, substring_ok=True, k=5):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    term = term.strip().lower()
    if substring_ok:
        for logit in logits:
            if logit == '':
                continue
            if term.startswith(logit):
                if len(logit) <= 2:
                    continue
                else:
                    return True
        return False
    else:
        return any(term in logit for logit in logits)

# in top/bot
def _is_say_x_feature(layer, feature_idx, word, feature_cache):
    feature_info = feature_cache[(layer, feature_idx)]
    if feature_info is None:
        return False
    return term_in_logits(word, feature_info['top_logits'], feature_info['bottom_logits'])

# fires directly on it
def _is_word_feature(layer, feature_idx, word, feature_cache, exclude='',):
    feature_info = feature_cache[(layer, feature_idx)]
    if feature_info is None:
        return False
    word_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        token = tokens[top_index]
        if (word in token) and token != exclude:
            word_count += 1
    return word_count >= 7

# it's within +- 2 tokens
def _is_near_word_feature(layer, feature_idx, word, feature_cache, exclude=''):
    feature_info = feature_cache[(layer, feature_idx)]
    if feature_info is None:
        return False
    word_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        s = max(top_index-2, 0)
        e = min(top_index + 2, len(tokens) - 1)
        if any((word in token or word[:2] == token[:2]) and token != exclude for token in tokens[s:e]):
            word_count += 1
    return word_count >= 7

#%%
# decide on which words to steer on
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

for model_name in models:
    models_and_transcoders = {
        'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
        'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
        'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
        'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
        'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
    }
    # get the activations to steer with

    metadata = pd.read_csv(f'../a_an/results/logit-lens/{model_name}/metadata.csv')

    model = ReplacementModel.from_pretrained(model_name, 
                                            models_and_transcoders['Qwen/' + model_name], 
                                            cpu_encoder=('8B' in model_name or '14B' in model_name), 
                                            dtype=torch.bfloat16)

    feature_info_cache = {}
    is_near_word_feature = lru_cache(maxsize=None)(
        partial(_is_near_word_feature, feature_cache=feature_info_cache)
    )
    is_word_feature = lru_cache(maxsize=None)(
        partial(_is_word_feature, feature_cache=feature_info_cache)
    )

    is_say_x_feature = lru_cache(maxsize=None)(
        partial(_is_say_x_feature, feature_cache=feature_info_cache)
    )

    chattify = partial(_chattify, tokenizer=model.tokenizer)
    get_topk = partial(_get_topk, tokenizer=model.tokenizer)

    get_features_with_cache = partial(_get_features_with_cache, cache=feature_info_cache, model_name=model_name)

    feature_set = set()
    # collect all the features
    for article, profession in zip(metadata['correct_articles'], metadata['professions']):
        graph = Graph.from_pt(f'../a_an/attribution_graphs/{model_name}/{article}-{profession}.pt')
        selected_acts = graph.active_features
        position_acts = selected_acts[selected_acts[:, 1] == graph.n_pos - 1]
        feature_set.update((layer, feat) for layer, _, feat in position_acts.tolist())

    get_features_with_cache(feature_set)
    
    name_to_features = {}
    for article, profession in zip(metadata['correct_articles'], metadata['professions']):
        graph = Graph.from_pt(f'../a_an/attribution_graphs/{model_name}/{article}-{profession}.pt')
        selected_acts = graph.active_features
        position_acts = selected_acts[selected_acts[:, 1] == graph.n_pos - 1]

        acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                        graph.activation_values, 
                                        size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))

        features_with_acts = [(layer, feat, acts[layer, pos, feat].item()) 
                                for layer, pos, feat in position_acts.tolist()
                                if  is_word_feature(layer, feat, profession) 
                                    or is_say_x_feature(layer, feat, profession)]
        name_to_features[profession] = features_with_acts

    professions_sorted = sorted(list(name_to_features.keys()), key=lambda k: len(name_to_features[k]), reverse=True)
    top10_professions = professions_sorted[:30]
    
    #%%
    # get the dataset to steer on
    ds = load_dataset("roneneldan/TinyStories")
    
    # Create results directory
    results_dir = Path("results/word_steering_tinystories_a_an_extended")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data for dataframe
    data_rows = []
    
    # %%
    for sentence_idx, sentence in enumerate(ds['train'].take(200)['text']):
        n_ctx = random.randint(5, 25)
        input_sentence = ' '.join(sentence.split(' ')[:n_ctx])

        original_generation = model.generate(input_sentence, do_sample=False)
        
        # Create base row data with 10 selected words
        base_data = {
            'sentence_index': sentence_idx,
            'input_text': input_sentence,
        }
        # Add row for original generation (no steering)
        original_row = base_data.copy()
        original_row.update({
            'generation': original_generation,
            'word_steered': None,
            'index_steered': None,
            'feature_count': None,
            'steering_strength': None
        })
        data_rows.append(original_row)
        
        # Generate with interventions - each gets its own row
        for i, n in enumerate(top10_professions):
            for steer_strength in [3,5,7]:
                interventions = [(l,slice(-1, None),f, steer_strength*a) for l,f,a in name_to_features[n]]
                generation, _, _ = model.feature_intervention_generate(chattify([input_sentence]), 
                                interventions, return_activations=False, do_sample=False)
                
                # Create row for this specific intervention
                intervention_row = base_data.copy()
                intervention_row.update({
                    'generation': generation,
                    'word_steered': n,
                    'index_steered': i,
                    'feature_count': len(interventions),
                    'steering_strength': steer_strength
                })
                data_rows.append(intervention_row)
    
    # Create dataframe and save
    df = pd.DataFrame(data_rows)
    output_file = results_dir / f"{model_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved results for {model_name} to {output_file}")

