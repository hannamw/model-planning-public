#%%
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm


def chattify(inputs: List[str], tokenizer):
    all_inputs = []
    for i, prompt in enumerate(inputs):
        all_inputs.append({'role': ('assistant' if i % 2 else 'user'), 'content': prompt})
    chattified = tokenizer.apply_chat_template(all_inputs, tokenize=False, add_generation_prompt=False)[:-11]
    if chattified.endswith('<|im_end|>\n'):
        chattified = chattified[:-len('<|im_end|>\n')]
    return chattified
#%%
model_name = "Qwen/Qwen3-14B"
    # Load model and tokenizer
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
#%%
first_line = "Golden echoes drift like dust through yesterday's dim light,"
prompt = chattify([f"/no_think Write only the next line of this rhyming couplet: {first_line.strip()}", ""], tokenizer)
    
# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.cuda() for k, v in inputs.items()}

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print(tokenizer.decode(outputs[0].tolist()))
#%%
prompt_template = ["/no_think Answer the following question in one word. Q: {question}", "<think>\n\n</think>\n\nA:"]
prompt_template_chattified = chattify(prompt_template, tokenizer)

question = 'The country containing Alexandria has its capital in which city?'

query = prompt_template_chattified.format(question=question)

output = model.generate(**tokenizer(query, return_tensors='pt'))
print(tokenizer.decode(output[0]))
# %%
prompt_template = ["Fact: {question}"]
prompt_template_chattified = chattify(prompt_template, tokenizer)

question = '<|im_start|>user\nFact: The state containing Alexandria has its capital in'
query = question
#query = prompt_template_chattified.format(question=question)

output = model.generate(**tokenizer(query, return_tensors='pt'))
print(tokenizer.decode(output[0]))
# %%
