#%%
# Import additional required modules
import re
import json
from pathlib import Path
from functools import lru_cache

import pandas as pd
from tqdm import tqdm
import torch

from circuit_tracer.graph import Graph
from circuit_tracer import ReplacementModel

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 1000
LAST_ONLY = True
INTERVENTION_RESULTS_DIR = Path('results/intervention_on_a_an_last') if LAST_ONLY else Path('results/intervention_on_a_an')  # Directory for path length results

models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]][-2:-1]
model_sizes = [28,28, 36, 36, 40]
#%%
# Mapping from model config names to logit lens names
model_to_logit_lens = {
    'qwen3-0.6b-relu-lowl0': 'Qwen3-0.6B',
    'qwen3-1.7b-relu-lowl0': 'Qwen3-1.7B',
    'qwen3-4b-relu': 'Qwen3-4B',
    'qwen3-8b-relu': 'Qwen3-8B',
    'qwen3-14b-relu-lowl0': 'Qwen3-14B'
}

logit_lens_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}


def load_top_logits(model_name, layer):
    target = logit_lens_to_transcoders[model_name].split("/")[-1]
    with open(f'../cache/top_logits/{target}-{layer}.json', 'r') as f:
        return json.load(f)

def term_in_logits(term:str, top: list[str], bottom: list[str], use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    term = term.strip().lower()
    if substring_ok:
        len1 = 0
        for logit in logits:
            if logit == '' or logit == 'a' or logit == 'an':
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

#%%
def load_model_results(model_name: str, results_dir: str = 'results/logit-lens'):
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    metadata_path = model_dir / 'metadata.csv'
    metadata = pd.read_csv(metadata_path)
    return metadata

@lru_cache(maxsize=100000)
def a_an_in_logits_count(top: tuple[str], bottom: tuple[str], count=2, use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    return sum((logit == 'a' or logit == 'an') for logit in logits) > count

#%%
# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
for model in models:
    print(f"Processing model: {model}")
    
    logit_lens_model_name = model
    replacement_model = ReplacementModel.from_pretrained('Qwen/' + logit_lens_model_name, 
                                                        logit_lens_to_transcoders[logit_lens_model_name], dtype=torch.bfloat16,
                                                        cpu_encoder=('14B' in logit_lens_model_name or '8B' in logit_lens_model_name))
    
    metadata = load_model_results(logit_lens_model_name)
    #metadata = metadata.head(10)
    
    graph_dir = Path('attribution_graphs') / model
    diff_graph_dir = Path('attribution_graphs_diff') / model
    
    # Add new columns to metadata for storing results
    metadata['original_a_prob'] = None
    metadata['original_an_prob'] = None
    metadata['zeroed_a_prob'] = None
    metadata['zeroed_an_prob'] = None
    metadata['multiplied_a_prob'] = None
    metadata['multiplied_an_prob'] = None
    metadata['selected_nodes_count'] = None

    top_bottom_by_layer = {}
    for layer in range(model.cfg.n_layers):
        top_bottom_by_layer[layer] = load_top_logits(logit_lens_model_name, layer)
    
    # Process each example based on metadata
    for idx, row in tqdm(metadata.iterrows()):
        correct_article = row['correct_articles']
        profession = row['professions']
        
        # Generate filename based on metadata
        graph_name = f"{correct_article}-{profession}"
        filename = f"{correct_article}-{profession}.pt"
        graph_file = graph_dir / filename
        diff_graph_file = diff_graph_dir / filename
        
        # Get important nodes for this example
        example_key = f"{correct_article}-{profession}"

        graph = Graph.from_pt(str(graph_file))

        n_pos = graph.n_pos 
        s = graph.input_string
        
        original_logits, original_acts = replacement_model.get_activations(s)

        # Get token IDs for 'a' and 'an'
        tokenizer = replacement_model.tokenizer
        a_token_id = tokenizer.encode(' a', add_special_tokens=False)[0]
        an_token_id = tokenizer.encode(' an', add_special_tokens=False)[0]
        
        # Extract probabilities for 'a' and 'an' tokens
        # Apply softmax to get probabilities
        original_probs = torch.softmax(original_logits[0, -1, :], dim=-1)
        
        # Get specific probabilities for 'a' and 'an'
        original_a_prob = original_probs[a_token_id].item()
        original_an_prob = original_probs[an_token_id].item()

        candidate_nodes = graph.active_features[graph.active_features[:, 1] == graph.n_pos - 1] if LAST_ONLY else graph.active_features

        a_an_feature_indices = [(layer, pos, feat) for layer, pos, feat in candidate_nodes.tolist() 
                                    if a_an_in_logits_count(tuple(top_bottom_by_layer[layer][0][feat]),
                                                            tuple(top_bottom_by_layer[layer][1][feat]), count=3)]
        
        # If we have selected nodes, perform interventions
        if a_an_feature_indices:
            #print(original_acts.size())
            #sn = torch.tensor(selected_nodes)
            #print(sn.max(0).values)
            zero_interventions = [(*feat, 0) for feat in a_an_feature_indices]
            multiply_interventions = [(*feat, 5 * original_acts[tuple(feat)]) for feat in a_an_feature_indices]
            logits_zeros, acts_zeros = replacement_model.feature_intervention(s, interventions=zero_interventions)
            logits_multiply, acts_multiply = replacement_model.feature_intervention(s, interventions=multiply_interventions)
            
            zerod_probs = torch.softmax(logits_zeros[0, -1, :], dim=-1)
            multiplied_probs = torch.softmax(logits_multiply[0, -1, :], dim=-1)

            zeroed_a_prob = zerod_probs[a_token_id].item()
            zeroed_an_prob = zerod_probs[an_token_id].item()
            multiplied_a_prob = multiplied_probs[a_token_id].item()
            multiplied_an_prob = multiplied_probs[an_token_id].item()
            selected_nodes_count = len(a_an_feature_indices)
        else:
            # No selected nodes - use original probabilities for interventions
            zeroed_a_prob = original_a_prob
            zeroed_an_prob = original_an_prob
            multiplied_a_prob = original_a_prob
            multiplied_an_prob = original_an_prob
            selected_nodes_count = 0
        
        # Store probabilities directly in metadata DataFrame
        metadata.at[idx, 'original_a_prob'] = original_a_prob
        metadata.at[idx, 'original_an_prob'] = original_an_prob
        metadata.at[idx, 'zeroed_a_prob'] = zeroed_a_prob
        metadata.at[idx, 'zeroed_an_prob'] = zeroed_an_prob
        metadata.at[idx, 'multiplied_a_prob'] = multiplied_a_prob
        metadata.at[idx, 'multiplied_an_prob'] = multiplied_an_prob
        metadata.at[idx, 'selected_nodes_count'] = selected_nodes_count
        
    # Save the metadata with intervention results as CSV
  

# %%
