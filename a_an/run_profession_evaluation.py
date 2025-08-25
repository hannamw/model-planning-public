#%%
import torch
from pathlib import Path
from evaluate_professions_wrong import evaluate_professions

model_names = [f'Qwen/Qwen3-{size}B' for size in ['0.6', '1.7', '4', '8', '14', '32']]

#%%
for article in ['a', 'an', None][2:]:
    for model_name in model_names:
        model_name_noslash = model_name.split('/')[-1]
        folder_article = 'random' if article is None else article
        output_dir = Path(f'results/a-an-IC-{folder_article}')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir /f'{model_name_noslash}.csv'
        dtype = torch.bfloat16# if '14' in model_name or '32' in model_name else torch.float32
        evaluate_professions(model_name=model_name, output_path=output_path, dtype=dtype, ic_article_filter=article)
        torch.cuda.empty_cache()
        #print(f'GPU allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB')
# %%
