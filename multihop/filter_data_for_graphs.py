#%%
import pandas as pd

#%%
data_df = pd.read_csv('data/combined_multihop_dataset.csv')
# question,intermediate,answer,intermediate_expression,prompt_type,prompt_subtype
per_model_dfs = {}
models = [f"Qwen3-{size}B" for size in [0.6, 1.7,4,8,14,32]]
for model in models:
    per_model_dfs[model] = pd.read_csv(f'results/behavioral/{model}.csv')
    # question,generated,answer,intermediate,prompt_type,exact_match,contains_answer

#%%
prompt_types = data_df['prompt_type'].unique()
selected_dfs = []

for prompt_type in prompt_types:
    type_df = data_df[data_df['prompt_type'] == prompt_type]
    if len(type_df) < 200:
        selected_dfs.append(type_df)
    else:
        # Get Qwen-32B results for this prompt type
        qwen32_results = per_model_dfs['Qwen3-32B'][per_model_dfs['Qwen3-32B']['question'].isin(type_df['question'])]
        
        # Split into correct and incorrect based on contains_answer
        correct = type_df[type_df['question'].isin(qwen32_results[qwen32_results['contains_answer']]['question'])]
        incorrect = type_df[type_df['question'].isin(qwen32_results[~qwen32_results['contains_answer']]['question'])]
        
        # Sample up to 100 from each category
        if len(correct) >= 100 and len(incorrect) >= 100:
            selected = pd.concat([correct.sample(100), incorrect.sample(100)])
        else:
            # If one category has less than 100, take all from that category and fill with the other
            if len(correct) < 100:
                selected = pd.concat([
                    correct,
                    incorrect.sample(200 - len(correct))
                ])
            else:
                selected = pd.concat([
                    correct.sample(200 - len(incorrect)),
                    incorrect
                ])
        selected_dfs.append(selected)

# Combine all selected data
final_df = pd.concat(selected_dfs, ignore_index=True)

# Add columns for each model's performance
for model in models:
    model_results = per_model_dfs[model]
    # Create a mapping of question to results
    exact_match_dict = dict(zip(model_results['question'], model_results['exact_match']))
    contains_answer_dict = dict(zip(model_results['question'], model_results['contains_answer']))
    
    # Add columns for this model
    final_df[f'{model}_exact_match'] = final_df['question'].map(exact_match_dict)
    final_df[f'{model}_contains_answer'] = final_df['question'].map(contains_answer_dict)

# Save the final dataframe
final_df.to_csv('data/filtered_multihop_dataset.csv', index=False)
# %%
