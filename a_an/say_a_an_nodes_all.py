#%%
# Import additional required modules
import re
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from circuit_tracer.graph import Graph

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 1000
LAST_ONLY = True
PATH_LENGTH_RESULTS_DIR = Path('results/directness_diff_last') if LAST_ONLY else Path('results/directness_diff')  # Directory for path length results

model_sizes = [28,28, 36, 36, 40]

logit_lens_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}
models = list(logit_lens_to_transcoders.keys())

#%%
def load_top_logits(model_name, layer):
    target = logit_lens_to_transcoders[model_name].split("/")[-1]
    with open(f'../cache/top_logits/{target}-{layer}.json', 'r') as f:
        return json.load(f)


def a_an_in_logits_count(top: list[str], bottom: list[str], count=2, use_bottom=True, substring_ok=True, k=10):
    logits = top[:k] + bottom[:k] if use_bottom else top[:k]
    # Preprocess logits: strip spaces and non-alphanumeric characters from the sides
    logits = [re.sub(r'^[^\w]+|[^\w]+$', '', logit.strip()).lower() for logit in logits]
    return sum((logit == 'a' or logit == 'an') for logit in logits) > count

#%%
# Store results for all models
# Load important nodes for all models
# The load_important_nodes function now handles loading and filtering
# We need to pass the model_name and example_key to it
for model in models:
    print(f"Processing model: {model}")
    a_an_counts = []
    a_an_features = []
    top_bottom_by_layer = {}
    
    for layer in range(60):
        a_an_features_layer = []
        try:
            top, bottom = load_top_logits(model, layer)
            top_bottom_by_layer[layer] = (top, bottom)
        except:
            print(f"Broke at {layer}")
            break
        
        for i, (top_i, bottom_i) in enumerate(zip(top, bottom)):
            is_a_an = a_an_in_logits_count(top_i, bottom_i, count=3)
            if is_a_an:
                a_an_features_layer.append(i)
        a_an_count = len(a_an_features_layer)
        a_an_counts.append(a_an_count)
        a_an_features.append(a_an_features_layer)

    print(sum(a_an_counts))

# %%
