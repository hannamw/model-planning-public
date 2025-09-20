#%%
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set

import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
#%%
def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified

model = HookedTransformer.from_pretrained_no_processing('Qwen/Qwen3-14B', 
                                                        dtype=torch.bfloat16)
#%%
first_line = 'In starlit whispers, my hopes take flight,'

tokenizer = model.tokenizer
max_tokens = 20

prompt = f"/no_think Write only the next line of this rhyming couplet: {first_line.strip()},"

chattified_prompt = chattify([prompt, ""], tokenizer)
# Tokenize input
inputs = tokenizer(chattified_prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

input_ids = inputs['input_ids']
# Generate
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    print(model.tokenizer.decode(outputs.squeeze(0)[input_ids.size(1):]))
# %%
