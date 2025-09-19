#%%
import re
import pandas as pd
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

all_model_dfs = []
for model in models:
    df = pd.read_csv(model + '.csv')
    for column in ['original_generation','intervention_generation','temp_0.3_sample_1','temp_0.3_sample_2','temp_0.3_sample_3','temp_0.3_sample_4','temp_0.3_sample_5','temp_0.7_sample_1','temp_0.7_sample_2','temp_0.7_sample_3','temp_0.7_sample_4','temp_0.7_sample_5','temp_1.0_sample_1','temp_1.0_sample_2','temp_1.0_sample_3','temp_1.0_sample_4','temp_1.0_sample_5']:
        df[column] = [re.sub('.*</think>\n\n', '', x, flags=re.DOTALL) for x in df[column]]
    for column in ['clean_completion','raw_completion','raw_completion_without_last']:
        del df[column]
    df.to_csv(model + '.csv')

# %%
