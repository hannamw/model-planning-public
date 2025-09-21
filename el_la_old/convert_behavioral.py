#%%
import pandas as pd

DEFAULT_QWEN3_MODELS = [
    "Qwen3-0.6B",
    "Qwen3-1.7B", 
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
]

dfs = {model: pd.read_csv(f'results/behavioral/{model}.csv') for model in DEFAULT_QWEN3_MODELS}
#%%
correct32 = dfs['Qwen3-32B']['Noun_Correct']
#%%
for model, df in dfs.items():
    filtered_df = df[correct32]
    filtered_df.to_csv(f'results/behavioral-filtered/{model}.csv')
# %%
