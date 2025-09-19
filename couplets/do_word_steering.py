#%%
import re
from collections import defaultdict
from functools import partial, lru_cache
import random

import torch
import pandas as pd
from datasets import load_dataset


from circuit_tracer import ReplacementModel, Graph
from circuit_tracer.utils.decode_url_features import pinned_ids_string
from load_feature_from_binary import get_features_top_acts_from_list

#%%
# decide on which words to steer on
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

model_name = models[-1]

data = torch.load(f'results/feature_counts/{model_name}.pt')

def get_word_count(feature_data, count_type):
    counts = feature_data[count_type]
    say_x_count = counts['say_x_feature_count'].numpy()[-2]
    word_count = counts['word_feature_count'].numpy()[-2]
    near_word_count = counts['near_word_feature_count'].numpy()[-2]
    return say_x_count + word_count + near_word_count


c = defaultdict(list)
for name, feature_data in data.items():
    _, last_word = name.split('-')
    c[last_word].append((name, get_word_count(feature_data, 'selected_counts')))

d = {k: tuple(list(x) for x in zip(*v)) for k, v in c.items()}

sorted_counts = sorted(d.items(), key=lambda x: sum(x[1][1]) / len(x[1][1]), reverse=True)
sorted_counts = [(x,y,z) for x, (y,z) in sorted_counts if all(0<count < 100 for count in z)]
print(sorted_counts)
top10 = sorted_counts[:10]
top10_names = []
for word, names, counts in top10:
    i = counts.index(max(counts))
    top10_names.append(names[i])
#%%

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
# get the activations to steer with

model = ReplacementModel.from_pretrained(model_name, 
                                        models_and_transcoders['Qwen/' + model_name], 
                                        cpu_encoder=('8B' in model_name or '14B' in model_name), 
                                        dtype=torch.bfloat16)

#%%
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

get_features_with_cache = partial(_get_features_with_cache, cache=feature_info_cache, model_name=model_name)

feature_set = set()
# collect all the features
for name in top10_names:
    graph = Graph.from_pt(f'attribution_graphs/{model_name}/{name}.pt')
    selected_acts = graph.active_features[graph.selected_features]
    position_acts = selected_acts[selected_acts[:, 1] == graph.n_pos - 2]
    feature_set.update((layer, feat) for layer, _, feat in position_acts.tolist())

get_features_with_cache(feature_set)
#%%
name_to_features = {}
for name in top10_names:
    idx, word = name.split('-')
    graph = Graph.from_pt(f'attribution_graphs/{model_name}/{name}.pt')
    selected_acts = graph.active_features[graph.selected_features]
    position_acts = selected_acts[selected_acts[:, 1] == graph.n_pos - 2]

    acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                    graph.activation_values, 
                                    size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))

    features_with_acts = [(layer, feat, acts[layer, pos, feat].item()) 
                            for layer, pos, feat in position_acts.tolist()
                            if is_near_word_feature(layer, feat, word) 
                                or is_word_feature(layer, feat, word) 
                                or is_say_x_feature(layer, feat, word)]
    name_to_features[name] = features_with_acts
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
chattify = partial(_chattify, tokenizer=model.tokenizer)
get_topk = partial(_get_topk, tokenizer=model.tokenizer)
#%%
strong_own_interventions = [(l,slice(-1, None),f, 3*a) for l,f,a in name_to_features['93-own']]


generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, do_sample=False)
print(generation)
#%%
strong_own_interventions = [(l,slice(-1, None),f, 2*a) for l,f,a in name_to_features['36-day']]


generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, do_sample=False)
print(generation)

#%%
strong_own_interventions = [(l,slice(-1, None),f, 5*a) for l,f,a in name_to_features['178-stride']]


generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, do_sample=False)
print(generation)

#%%
strong_own_interventions = [(l,slice(-1, None),f, 2*a) for l,f,a in name_to_features['87-dream']]


generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, do_sample=False)
print(generation)
#%%
strong_own_interventions = [(l,slice(-1, None),f, 3*a) for l,f,a in name_to_features['91-night']]


generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                        strong_own_interventions, return_activations=False, do_sample=False)
print(generation)
#%%
for n in ['92-age', '115-unknown', '31-sun', '153-same', '38-light']:
    strong_own_interventions = [(l,slice(-1, None),f, 5*a) for l,f,a in name_to_features[n]]


    generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog"]), 
                            strong_own_interventions, return_activations=False, do_sample=False)
    print(n, generation)

#%%
model.generate(chattify(["Yesterday I saw a dog"]), do_sample=False)
#%%
logits = model(chattify(["Yesterday I saw a dog"])).squeeze(0)
probs = logits[-1].softmax(-1)
print(probs @ probs.log() * -1)
#%%
strong_own_interventions = [(l,slice(-1, None),f, 8*a) for l,f,a in name_to_features['153-same']]


int_logits, _ = model.feature_intervention(chattify(["Yesterday I saw a dog"]), 
                strong_own_interventions, return_activations=False)
int_probs = int_logits.squeeze(0)[-1].softmax(-1)
print(int_probs @ int_probs.log() * -1)
#%%
generation, gen_logits, _ = model.feature_intervention_generate(chattify(["Yesterday I saw a dog that"]), 
                            strong_own_interventions, return_activations=False, do_sample=False)
#%%
model.generate(chattify(["Yesterday I saw a dog that was"]), do_sample=True)
#%%
# get the dataset to steer on
ds = load_dataset("roneneldan/TinyStories")
# %%
for sentence in ds['train'].take(100)['text']:
    n_ctx = random.randint(5, 10)
    input_sentence = ' '.join(sentence.split(' ')[:n_ctx])
    for n in name_to_features.keys():
        for steer_strength in [3,5,7]:
            _interventions = [(l,slice(-1, None),f, steer_strength*a) for l,f,a in name_to_features[n]]
            generation, _, _ = model.feature_intervention_generate(chattify(["input_sentence"]), 
                            strong_own_interventions, return_activations=False, do_sample=False)

