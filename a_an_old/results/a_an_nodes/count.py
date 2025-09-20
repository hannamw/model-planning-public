#%%
import pandas as pd
#%%
dfs = [pd.read_csv(f"Qwen3-{size}B.csv") for size in [0.6, 1.7, 4, 8, 14]]
# %%
for df in dfs:
    print(df['a_an_counts'].mean())
# %%
