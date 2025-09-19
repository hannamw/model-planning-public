#%%
import torch
from pathlib import Path
from evaluate_professions_repeat import evaluate_professions

#%%
model_names = [f'Qwen/Qwen3-{size}B' for size in ['0.6', '1.7', '4', '8', '14', '32']]

#%%
for model_name in model_names:
    model_name_noslash = model_name.split('/')[-1]
    output_path = Path('results/a-an-repeat')/f'{model_name_noslash}.csv'
    dtype = torch.bfloat16 if '14' in model_name or '32' in model_name else torch.float32
    evaluate_professions(model_name=model_name, output_path=output_path, dtype=dtype)
# %%
