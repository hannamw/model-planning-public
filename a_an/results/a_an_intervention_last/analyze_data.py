#%%
import pandas as pd
import matplotlib.pyplot as plt
#%%
dfs = {model: pd.read_csv(model + '.csv') for model in [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]}
#%%
for model, df in dfs.items():
    print(model)
    print(df['say_a_an_node_zeroed_activation_diff'].mean())
    print(df['say_a_an_node_multiplied_activation_diff'].mean())
#%%