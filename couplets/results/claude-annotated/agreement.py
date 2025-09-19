#%%
import pandas as pd
#%% 
human = pd.read_csv('Qwen3-14B_analyzed_annotated.csv', delimiter=';')
claude = pd.read_csv('Qwen3-14B_analyzed.csv')
#%%
claude = claude.head(len(human))
#%%
col = 'contains_steered_word'
(human[col] == claude[col]).mean()
#%%
col = 'is_coherent'
(human[col] == claude[col]).mean()
#%%
col = 'adapted_context'
(human[col] == claude[col]).mean()
#%%
col = 'is_coherent'
(human[col]).mean(), (claude[col]).mean()
#%%
col = 'adapted_context'
(human[col]).mean(),  claude[col].mean()
#%%
cols = ['contains_steered_word', 'is_coherent', 'adapted_context']
(human[cols].eq(claude[cols]).all(axis=1)).mean()
# %%
((human[cols[0]] & human[cols[1]] & human[cols[2]]) == (claude[cols[0]] & claude[cols[1]] & claude[cols[2]])).mean()
# %%
(~(human[cols[0]] & human[cols[1]] & human[cols[2]]) & (claude[cols[0]] & claude[cols[1]] & claude[cols[2]])).mean()
# %%
