#%%
from pathlib import Path
from functools import partial
from collections import defaultdict

import pandas as pd
import torch

from transformer_lens import HookedTransformer
from circuit_tracer import Graph

intervention_range = range(18, 20)
coeff = 4.0

models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
#%%
for model_name in models:
    whole_model_name = f"Qwen/{model_name}"
    model = HookedTransformer.from_pretrained_no_processing(whole_model_name, 
                                             dtype=torch.bfloat16, )

    graph_dir = Path(f'attribution_graphs/{model_name}')
    metadata = pd.read_csv(f'results/attribution_metadata/{model_name}.csv', index_col=0)
    act_dict = defaultdict(list)

    for idx, row in metadata.iterrows():
        second_last_word = row['second_last_word']
        rhyme_group = row['rhyme_group']
        graph = Graph.from_pt(graph_dir / f"{idx}-{second_last_word}.pt")
        input_tokens = [model.tokenizer.decode(token) for token in graph.input_tokens]
        im_end_idx = input_tokens.index('<|im_end|>')
        input_string = model.tokenizer.decode(graph.input_tokens)
        _, cache = model.run_with_cache(input_string)

        for layer in range(model.cfg.n_layers):
            steering_act = cache[f'blocks.{layer}.hook_mlp_out'].squeeze(0)[im_end_idx - 2]
            act_dict[(rhyme_group, layer)].append(steering_act)

    act_dict = {k: torch.stack(v).mean(0) for k,v in act_dict.items()}

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
        valid_rows = metadata[metadata['rhyme_group'] != rhyme_group]
        chosen_row_raw = valid_rows.sample(1)
        chosen_row = chosen_row_raw.iloc[0]
        chosen_index = chosen_row_raw.index[0]
        chosen_rhyme_group = chosen_row['rhyme_group']
        chosen_second_last_word = chosen_row['second_last_word']
        chosen_id = f'{chosen_index}-{chosen_second_last_word}'
        
        # Store successful intervention metadata
        metadata.at[idx, 'found_valid_row'] = True
        metadata.at[idx, 'chosen_id'] = chosen_id
        metadata.at[idx, 'chosen_index'] = chosen_index
        metadata.at[idx, 'chosen_rhyme_group'] = chosen_rhyme_group

        intervention_dict = {f'blocks.{layer}.hook_mlp_out' :act_dict[(chosen_rhyme_group, layer)] - act_dict[(rhyme_group, layer)] 
                                for layer in range(model.cfg.n_layers)}

        def steering_intervention(acts, hook):
            return acts + coeff * intervention_dict[hook.name]

        steering_interventions = [(f'blocks.{layer}.hook_mlp_out', steering_intervention) for layer in intervention_range]

        with model.hooks(fwd=steering_interventions):
            new_generation = model.generate(
                input_text, 
                do_sample=False, 
                max_new_tokens=20
            )
        
        # Store intervention generation
        metadata.at[idx, 'intervention_generation'] = new_generation
        
    # Ensure output directory exists
    Path('results/rhyme_intervention_dim').mkdir(parents=True, exist_ok=True)
    metadata.to_csv(f'results/rhyme_intervention_dim/{model_name}.csv')
# %%
