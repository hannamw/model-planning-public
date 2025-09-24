#%%
from pathlib import Path
import re
import string
from collections import Counter
from functools import partial, lru_cache

import pandas as pd
import torch

from circuit_tracer import Graph, ReplacementModel
from load_feature_from_binary import get_features_top_acts_from_list

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

def _is_rhyme_feature(layer, feature_idx, word, feature_cache, exclude=None):
    feature_info = feature_cache[(layer, feature_idx)]
    first_chars, last_chars, token_counts = Counter(), Counter(), Counter()
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
    top_true = (token_count <= 5) and (exclude is None or exclude not in most_common_token) and ((most_common_first_char in 'aeiou' and first_count >=7) or last_count >=7)
    return top_true

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
for model_name in models[-1:]:
    feature_info_cache = {}

    is_rhyme_word_feature = lru_cache(maxsize=None)(
        partial(_is_rhyme_feature, feature_cache=feature_info_cache)
    )

    whole_model_name = f"Qwen/{model_name}"
    transcoders_name = models_and_transcoders[whole_model_name]
    model = ReplacementModel.from_pretrained(whole_model_name, 
                                             transcoders_name,
                                             dtype=torch.bfloat16)

    def chattify(inputs: list[str]):
        all_inputs = []
        for i, prompt in enumerate(inputs):
            all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
        chattified = model.tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
        if chattified.endswith('<|im_end|>\n'):
            chattified = chattified[:-len('<|im_end|>\n')]
        return chattified


    graph_dir = Path(f'attribution_graphs/{model_name}')
    metadata = pd.read_csv(f'results/rhyme_intervention_sample/{model_name}.csv', index_col=0)

    for column in 'intervention_temp_1.0_sample_1,intervention_temp_1.0_sample_2,intervention_temp_1.0_sample_3,intervention_temp_1.0_sample_4,intervention_temp_1.0_sample_5,temp_0.3_sample_1,temp_0.3_sample_2,temp_0.3_sample_3,temp_0.3_sample_4,temp_0.3_sample_5,temp_0.7_sample_1,temp_0.7_sample_2,temp_0.7_sample_3,temp_0.7_sample_4,temp_0.7_sample_5,temp_1.0_sample_1,temp_1.0_sample_2,temp_1.0_sample_3,temp_1.0_sample_4,temp_1.0_sample_5'.split(','):
        del metadata[column]

    feature_set = set()

    id_to_features_acts = {}
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")

        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)

        last_word_features = graph.active_features[graph.active_features[:, 1] == last_word - 2]
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

        last_word_features = graph.active_features[graph.active_features[:, 1] == last_word - 2]
        last_word_features_unique = list(set((layer, feature) for layer, _, feature in last_word_features.tolist()))
        last_word_feature_infos = get_features_with_cache(last_word_features_unique, feature_info_cache, model_name)
        rhyme_features = {k:v for k,v in last_word_feature_infos.items() if is_rhyme_word_feature(*k, second_last_word)}
        
        id_to_features_acts[f"{idx}-{second_last_word}"] = [(layer, feat, acts[layer, last_word - 2, feat]) 
                                                            for layer, feat in rhyme_features.keys()]

    metadata['feature_count'] = [len(id_to_features_acts[f"{idx}-{second_last_word}"] )
                                for idx, second_last_word in zip(metadata.index, metadata['second_last_word'])]

    # Initialize columns for intervention results
    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        rhyme_group = row['rhyme_group']
        id = f"{idx}-{second_last_word}"
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = model.tokenizer.convert_ids_to_tokens(graph.input_tokens)
        last_word = input_tokens.index(model.tokenizer.eos_token)


        chosen_id = metadata.at[idx, 'chosen_id']
        acts = torch.sparse_coo_tensor(graph.active_features.t(), 
                                        graph.activation_values, 
                                        size=(graph.cfg.n_layers, graph.n_pos, model.transcoders.d_transcoder))
        
        downweight_interventions = [(layer, last_word - 2, feature, -3 * act) 
                                for layer, feature, act in id_to_features_acts[id]]
        new_word_interventions = [(layer, last_word - 2, feature, 7 * act) 
                                for layer, feature, act in id_to_features_acts[chosen_id]]

        # Generate original text
        unthink_idx = input_tokens.index('</think>')
        input_text = model.tokenizer.decode(graph.input_tokens[:unthink_idx + 2])
        original_generation_stripped = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', metadata.at[idx, 'clean_completion'])
        og_split = original_generation_stripped.split()
        og_to_last, og_last = ' '.join(og_split[:-1]), og_split[-1]
        
        # Store intervention generation (strip input text)
        new_generation_stripped = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', metadata.at[idx, 'intervention_generation'])
        new_split = new_generation_stripped.split()
        new_to_last, new_last = ' '.join(new_split[:-1]), new_split[-1]

        orig_gen_intervention, _, _ = model.feature_intervention_generate(input_text + og_to_last, 
                                                                     interventions=downweight_interventions + new_word_interventions, 
                                                                     do_sample=False,
                                                                     max_new_tokens=2,
                                                                    )
        orig_gen_intervention_stripped = re.sub(f'.*{og_to_last}', '', orig_gen_intervention, flags=re.DOTALL).strip().split()[0]

        new_gen_no_intervention = model.generate(input_text + new_to_last, do_sample=False, max_new_tokens=2)
        new_gen_no_intervention_stripped = re.sub(f'.*{new_to_last}', '', new_gen_no_intervention, flags=re.DOTALL).strip().split()[0]

        original_no_intervention = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', og_last)
        original_intervention = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', orig_gen_intervention_stripped)
        new_no_intervention = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', new_gen_no_intervention_stripped)
        new_intervention = re.sub(r'^[{}]+|[{}]+$'.format(re.escape(string.punctuation), re.escape(string.punctuation)), '', new_last)

        og_to_last_logits = model(chattify([og_to_last]))
        new_to_last_logits = model(chattify([new_to_last]))
        
        # Get top-1 token identities and probabilities
        og_probs = torch.softmax(og_to_last_logits.squeeze(0)[-1], dim=-1)
        new_probs = torch.softmax(new_to_last_logits.squeeze(0)[ -1], dim=-1)
        
        og_top1_token_id = torch.argmax(og_to_last_logits.squeeze(0)[-1]).item()
        new_top1_token_id = torch.argmax(new_to_last_logits.squeeze(0)[-1]).item()
        
        og_top1_token = model.tokenizer.decode([og_top1_token_id])
        new_top1_token = model.tokenizer.decode([new_top1_token_id])
        
        # Get probabilities and logprobs for the actual last words
        og_last_token_ids = model.tokenizer.encode(' ' + og_last, add_special_tokens=False)
        new_last_token_ids = model.tokenizer.encode(' ' + new_last, add_special_tokens=False)
        
        # For multi-token words, we'll use the first token's probability
        og_last_token_id = og_last_token_ids[0] if og_last_token_ids else None
        new_last_token_id = new_last_token_ids[0] if new_last_token_ids else None
        
        og_last_prob = og_probs[og_last_token_id].item() if og_last_token_id is not None else 0.0
        new_last_prob = new_probs[new_last_token_id].item() if new_last_token_id is not None else 0.0
        
        og_last_logprob = torch.log_softmax(og_to_last_logits.squeeze(0)[-1], dim=-1)[og_last_token_id].item() if og_last_token_id is not None else float('-inf')
        new_last_logprob = torch.log_softmax(new_to_last_logits.squeeze(0)[-1], dim=-1)[new_last_token_id].item() if new_last_token_id is not None else float('-inf')
        
        # Add token data to existing metadata dataframe
        metadata.at[idx, 'og_to_last'] = og_to_last
        metadata.at[idx, 'og_last'] = og_last
        metadata.at[idx, 'new_to_last'] = new_to_last
        metadata.at[idx, 'new_last'] = new_last
        metadata.at[idx, 'original_no_intervention'] = original_no_intervention
        metadata.at[idx, 'original_intervention'] = original_intervention
        metadata.at[idx, 'new_no_intervention'] = new_no_intervention
        metadata.at[idx, 'new_intervention'] = new_intervention
        
        # Add top-1 token identities and probabilities/logprobs
        metadata.at[idx, 'og_top1_token'] = og_top1_token
        metadata.at[idx, 'new_top1_token'] = new_top1_token
        metadata.at[idx, 'og_last_prob'] = og_last_prob
        metadata.at[idx, 'new_last_prob'] = new_last_prob
        metadata.at[idx, 'og_last_logprob'] = og_last_logprob
        metadata.at[idx, 'new_last_logprob'] = new_last_logprob
    # Ensure output directory exists
    Path('results/rhyme_intervention_compare_contexts').mkdir(parents=True, exist_ok=True)
    metadata.to_csv(f'results/rhyme_intervention_compare_contexts/{model_name}.csv')
    del model
    torch.cuda.empty_cache()
# %%
