#%%
import re
from functools import lru_cache, partial
from collections import Counter, defaultdict
import requests

from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer

from circuit_tracer import Graph
from circuit_tracer.graph import prune_graph

from load_feature_from_binary import get_features_top_acts_from_list
#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

@lru_cache(maxsize=10000)
def fetch_rhymes(word: str) -> frozenset[str]:
    """Return a set of rhyming words for *word* (cached)."""
    word = word.lower()
    try:
        resp = requests.get("https://api.datamuse.com/words", params={"rel_rhy": word, "max": 1000}, timeout=20)
        resp.raise_for_status()
        return frozenset(item["word"].lower() for item in resp.json())
    except Exception as exc:  # noqa: BLE001  (broad ≈ network)
        print(f"Datamuse query failed for '{word}': {exc}")
        return frozenset()

def _is_EOL_feature(layer, feature_idx, feature_cache):
    feature_info = feature_cache[(layer, feature_idx)]
    EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        if top_index + 1 < len(tokens) and '⏎' in tokens[top_index + 1]:
            EOL_count += 1
    return EOL_count >= 7

def _is_near_EOL_feature(layer, feature_idx, feature_cache):
    feature_info = feature_cache[(layer, feature_idx)]
    near_EOL_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        near_EOL_tokens = tokens[top_index + 2: top_index + 5]
        if any('⏎' in tok for tok in near_EOL_tokens):
            near_EOL_count += 1
    return near_EOL_count >= 7

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

# in top/bot
def _is_say_x_feature(layer, feature_idx, word, feature_cache):
    feature_info = feature_cache[(layer, feature_idx)]
    return term_in_logits(word, feature_info['top_logits'], feature_info['bottom_logits'])

# fires directly on it
def _is_word_feature(layer, feature_idx, word, feature_cache, exclude='',):
    feature_info = feature_cache[(layer, feature_idx)]
    word_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        token = tokens[top_index]
        if (word in token or word[:2] == token[:2]) and token != exclude:
            word_count += 1
    return word_count >= 7

# it's within +- 2 tokens
def _is_near_word_feature(layer, feature_idx, word, feature_cache, exclude=''):
    feature_info = feature_cache[(layer, feature_idx)]
    word_count = 0
    for tokens, top_index in zip(feature_info['tokens'], feature_info['top_indices']):
        s = max(top_index-2, 0)
        e = min(top_index + 2, len(tokens) - 1)
        if any((word in token or word[:2] == token[:2]) and token != exclude for token in tokens[s:e]):
            word_count += 1
    return word_count >= 7

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

def setup_cache_and_counting(model_name: str):
    feature_info_cache = {}
    is_near_word_feature = lru_cache(maxsize=None)(
        partial(_is_near_word_feature, feature_cache=feature_info_cache)
    )
    is_word_feature = lru_cache(maxsize=None)(
        partial(_is_word_feature, feature_cache=feature_info_cache)
    )
    is_rhyme_feature = lru_cache(maxsize=None)(
        partial(_is_rhyme_feature, feature_cache=feature_info_cache)
    )
    is_say_x_feature = lru_cache(maxsize=None)(
        partial(_is_say_x_feature, feature_cache=feature_info_cache)
    )
    is_EOL_feature = lru_cache(maxsize=None)(
        partial(_is_EOL_feature, feature_cache=feature_info_cache)
    )
    is_near_EOL_feature = lru_cache(maxsize=None)(
        partial(_is_near_EOL_feature, feature_cache=feature_info_cache)
    )
    
    def record_counts(candidate_features, n_pos, word, rhyme_pos):
        counts = defaultdict(list)

        all_unique_features = list(set((layer, feature) for layer, _, feature in candidate_features.tolist()))
        get_features_with_cache(all_unique_features, feature_info_cache, model_name, verbose=True)


        pos_features = candidate_features[candidate_features[:, 1] == rhyme_pos]
        pos_features_unique = list(set((layer, feature) for layer, _, feature in pos_features.tolist()))
        pos_feature_infos = get_features_with_cache(pos_features_unique, feature_info_cache, model_name)
        rhyme_features = set((layer, feature_idx) for layer, feature_idx in pos_feature_infos.keys() if is_rhyme_feature(layer, feature_idx, word))

        pos_features = candidate_features[candidate_features[:, 1] == n_pos - 1]
        pos_features_unique = list(set((layer, feature) for layer, _, feature in pos_features.tolist()))
        pos_feature_infos = get_features_with_cache(pos_features_unique, feature_info_cache, model_name)
        last_rhyme_features = set((layer, feature_idx) for layer, feature_idx in pos_feature_infos.keys() if is_rhyme_feature(layer, feature_idx, word))

        for i in range(n_pos):
            pos_features = candidate_features[candidate_features[:, 1] == i]
            pos_features_unique = list(set((layer, feature) for layer, _, feature in pos_features.tolist()))
            pos_feature_infos = get_features_with_cache(pos_features_unique, feature_info_cache, model_name)

            eol_feature_count = sum(is_EOL_feature(layer, feature_idx) for layer, feature_idx in pos_feature_infos.keys())
            near_eol_feature_count = sum(is_near_EOL_feature(layer, feature_idx) for layer, feature_idx in pos_feature_infos.keys())
            say_x_feature_count = sum(is_say_x_feature(layer, feature_idx, word) for layer, feature_idx in pos_feature_infos.keys())
            word_feature_count = sum(is_word_feature(layer, feature_idx, word) for layer, feature_idx in pos_feature_infos.keys())
            near_word_feature_count = sum(is_near_word_feature(layer, feature_idx, word) for layer, feature_idx in pos_feature_infos.keys())
            rhyme_feature_count = sum(is_rhyme_feature(layer, feature_idx, word) for layer, feature_idx in pos_feature_infos.keys())
            specific_rhyme_feature_count = sum(((layer, feature_idx) in rhyme_features) for layer, feature_idx in pos_feature_infos.keys())
            last_rhyme_feature_count = sum(((layer, feature_idx) in last_rhyme_features) for layer, feature_idx in pos_feature_infos.keys())

            counts['eol_feature_count'].append(eol_feature_count)
            counts['near_eol_feature_count'].append(near_eol_feature_count)
            counts['say_x_feature_count'].append(say_x_feature_count)
            counts['word_feature_count'].append(word_feature_count)
            counts['near_word_feature_count'].append(near_word_feature_count)
            counts['rhyme_feature_count'].append(rhyme_feature_count)
            counts['specific_rhyme_feature_count'].append(specific_rhyme_feature_count)
            counts['last_rhyme_feature_count'].append(last_rhyme_feature_count)
        # convert to pt 
        counts = {k: torch.tensor(v) for k, v in counts.items()}
        return counts
    return feature_info_cache, record_counts

for model_name in models:
    print("Processing model:", model_name)
    feature_info_cache = {}
    whole_model_name = f"Qwen/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(whole_model_name)

    graph_dir = Path(f'attribution_graphs/{model_name}')
    metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)

    feature_info_cache, record_counts = setup_cache_and_counting(model_name)

    feature_set = set()
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        for layer, _, feature in graph.active_features.tolist():
            feature_set.add((layer, feature))

    get_features_with_cache(list(feature_set), feature_info_cache, model_name, verbose=True)

    to_save = {}
    for idx, row in metadata.iterrows():
        print("Processing example:", idx)
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        str_input_tokens = [tokenizer.decode(tok) for tok in graph.input_tokens]
        EOS_idx = str_input_tokens.index('<|im_end|>')

        print(f"Fetching {len(graph.active_features)} features")
        counts = record_counts(graph.active_features, graph.n_pos, second_last_word, EOS_idx - 2)
        print("finished first count")
        selected_counts = record_counts(graph.active_features[graph.selected_features], graph.n_pos, second_last_word, EOS_idx - 2)
        print("Finished second count")
        node_mask, _, _ = prune_graph(graph, node_threshold=0.8, edge_threshold=0.98)
        pruned_counts = record_counts(graph.active_features[graph.selected_features][node_mask[:len(graph.selected_features)]], graph.n_pos, second_last_word, EOS_idx - 2)

        to_save[f"{idx}-{second_last_word}"] = {
            'input_tokens': graph.input_tokens,
            'input_tokens_str': [tokenizer.decode(token) for token in graph.input_tokens],
            'counts': counts,
            'selected_counts': selected_counts,
            'pruned_counts': pruned_counts
        }

    torch.save(to_save, f'results/feature_counts/{model_name}.pt')
# %%
