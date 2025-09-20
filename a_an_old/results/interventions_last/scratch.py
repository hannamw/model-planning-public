#%%
import pandas as pd
#%%
df = pd.read_csv('Qwen3-14B.csv')
# %%
df[df['correct_articles'] == 'an']['selected_nodes_count'].mean()
# %%
df[df['correct_articles'] == 'a']['selected_nodes_count'].mean()
# %%
