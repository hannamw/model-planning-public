#%%
import re

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

# Keep a list of all model dataframes with metrics
all_model_dfs = []
for model in models:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/' + model)
    print(model)
    df = pd.read_csv(model + '.csv')
    
    # Add columns for correctness metrics and vowel/consonant detection
    df['model'] = model
    df['tokens_same'] = None
    df['orig_len'] = None
    df['intervened_len'] = None
    
    for idx, row in df.iterrows():
        original_generation = re.sub('.*</think>\n\n', '', row['original_generation'], flags=re.DOTALL).strip('.')
        intervention_generation = re.sub('.*</think>\n\n', '', row['intervention_generation'], flags=re.DOTALL).strip('.')
        
        original_tokens = tokenizer(original_generation, return_tensors='pt').input_ids.squeeze(0)
        intervention_tokens = tokenizer(intervention_generation, return_tensors='pt').input_ids.squeeze(0)
        min_len = min(original_tokens.size(0), intervention_tokens.size(0))
        matches = (original_tokens[:min_len] == intervention_tokens[:min_len])
        match_len = (matches.cumprod(dim=0)).sum().item()
        
        df.loc[idx, 'tokens_same'] = match_len
        df.loc[idx, 'orig_len'] = original_tokens.size(0)
        df.loc[idx, 'intervened_len'] = intervention_tokens.size(0)

        print("Original:", original_generation)
        print("Intervention:", intervention_generation)
        print("Tokens same:", df.loc[idx, 'tokens_same'])
        print("Original length:", df.loc[idx, 'orig_len'])
        print("Intervention length:", df.loc[idx, 'intervened_len'])
        print("--------------------------------")
    
    # Add this model's dataframe to the list
    all_model_dfs.append(df)

#%%
# Combine all model dataframes
combined_df = pd.concat(all_model_dfs, ignore_index=True)

# Calculate metrics for each model
model_metrics = []
for model in models:
    model_data = combined_df[combined_df['model'] == model]
    
    # Calculate vowel & consonant correct rate
    vowel_consonant_correct = (model_data['vowel_correct'] & model_data['consonant_correct']).mean()
    
    # Calculate vowel & singular consonant correct rate  
    vowel_singular_consonant_correct = (model_data['vowel_correct'] & model_data['singular_consonant_correct']).mean()
    
    # Calculate individual rates
    vowel_correct = model_data['vowel_correct'].mean()
    consonant_correct = model_data['consonant_correct'].mean()
    singular_consonant_correct = model_data['singular_consonant_correct'].mean()
    
    # Calculate tokens_same metrics
    mean_tokens_same = model_data['tokens_same'].mean()
    tokens_same_pct_orig = (model_data['tokens_same'] / model_data['orig_len']).mean()
    tokens_same_pct_intervened = (model_data['tokens_same'] / model_data['intervened_len']).mean()
    min_len = model_data[['orig_len', 'intervened_len']].min(axis=1)
    tokens_same_pct_min = (model_data['tokens_same'] / min_len).mean()
    
    model_metrics.append({
        'model': model,
        'vowel_consonant_correct': vowel_consonant_correct,
        'vowel_singular_consonant_correct': vowel_singular_consonant_correct,
        'vowel_correct': vowel_correct,
        'consonant_correct': consonant_correct,
        'singular_consonant_correct': singular_consonant_correct,
        'mean_tokens_same': mean_tokens_same,
        'tokens_same_pct_orig': tokens_same_pct_orig,
        'tokens_same_pct_intervened': tokens_same_pct_intervened,
        'tokens_same_pct_min': tokens_same_pct_min
    })

metrics_df = pd.DataFrame(model_metrics)

# Plot 1: Mean tokens_same by model
plt.figure(figsize=(10, 6))
x_positions = range(len(models))

plt.plot(x_positions, metrics_df['mean_tokens_same'], 
         marker='o', linewidth=2, markersize=8, 
         label='Mean Tokens Same', color='blue')

plt.xlabel('Model')
plt.ylabel('Mean Tokens Same')
plt.title('Mean Tokens Same by Model')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tokens_same_mean.pdf")
plt.show()

# Plot 2: Tokens same as percentage of different length measures
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Percentage of original length
axes[0].plot(x_positions, metrics_df['tokens_same_pct_orig'], 
             marker='o', linewidth=2, markersize=8, color='green')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Tokens Same / Original Length')
axes[0].set_title('Tokens Same as % of Original Length')
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(models, rotation=45)
axes[0].grid(True, alpha=0.3)

# Percentage of intervention length
axes[1].plot(x_positions, metrics_df['tokens_same_pct_intervened'], 
             marker='s', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Tokens Same / Intervention Length')
axes[1].set_title('Tokens Same as % of Intervention Length')
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(models, rotation=45)
axes[1].grid(True, alpha=0.3)

# Percentage of minimum length
axes[2].plot(x_positions, metrics_df['tokens_same_pct_min'], 
             marker='^', linewidth=2, markersize=8, color='purple')
axes[2].set_xlabel('Model')
axes[2].set_ylabel('Tokens Same / Min Length')
axes[2].set_title('Tokens Same as % of Min Length')
axes[2].set_xticks(x_positions)
axes[2].set_xticklabels(models, rotation=45)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tokens_same_percentages.pdf")
plt.show()

# Plot 3: Conditioned on vowel_consonant_correct
# First add the vowel_consonant_correct column to the combined dataframe
combined_df['vowel_consonant_correct'] = combined_df['vowel_correct'] & combined_df['consonant_correct']

# Calculate metrics conditioned on vowel_consonant_correct
conditioned_metrics = []
for model in models:
    model_data = combined_df[combined_df['model'] == model]
    
    for condition in [True, False]:
        condition_data = model_data[model_data['vowel_consonant_correct'] == condition]
        
        if len(condition_data) > 0:
            mean_tokens_same = condition_data['tokens_same'].mean()
            tokens_same_pct_orig = (condition_data['tokens_same'] / condition_data['orig_len']).mean()
            tokens_same_pct_intervened = (condition_data['tokens_same'] / condition_data['intervened_len']).mean()
            min_len = condition_data[['orig_len', 'intervened_len']].min(axis=1)
            tokens_same_pct_min = (condition_data['tokens_same'] / min_len).mean()
            count = len(condition_data)
        else:
            mean_tokens_same = 0
            tokens_same_pct_orig = 0
            tokens_same_pct_intervened = 0
            tokens_same_pct_min = 0
            count = 0
        
        conditioned_metrics.append({
            'model': model,
            'vowel_consonant_correct': condition,
            'mean_tokens_same': mean_tokens_same,
            'tokens_same_pct_orig': tokens_same_pct_orig,
            'tokens_same_pct_intervened': tokens_same_pct_intervened,
            'tokens_same_pct_min': tokens_same_pct_min,
            'count': count
        })

conditioned_df = pd.DataFrame(conditioned_metrics)

# Plot conditioned results
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Mean tokens same
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[0, 0].plot(x_positions, condition_data['mean_tokens_same'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=label, color=color)

axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Mean Tokens Same')
axes[0, 0].set_title('Mean Tokens Same by Model (Conditioned)')
axes[0, 0].set_xticks(x_positions)
axes[0, 0].set_xticklabels(models, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Percentage of original length
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[0, 1].plot(x_positions, condition_data['tokens_same_pct_orig'], 
                    marker='s', linewidth=2, markersize=8, 
                    label=label, color=color)

axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Tokens Same / Original Length')
axes[0, 1].set_title('Tokens Same as % of Original Length (Conditioned)')
axes[0, 1].set_xticks(x_positions)
axes[0, 1].set_xticklabels(models, rotation=45)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Percentage of intervention length
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[1, 0].plot(x_positions, condition_data['tokens_same_pct_intervened'], 
                    marker='^', linewidth=2, markersize=8, 
                    label=label, color=color)

axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Tokens Same / Intervention Length')
axes[1, 0].set_title('Tokens Same as % of Intervention Length (Conditioned)')
axes[1, 0].set_xticks(x_positions)
axes[1, 0].set_xticklabels(models, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Percentage of minimum length
for i, condition in enumerate([True, False]):
    condition_data = conditioned_df[conditioned_df['vowel_consonant_correct'] == condition]
    label = f'Vowel & Consonant Correct: {condition}'
    color = 'green' if condition else 'red'
    axes[1, 1].plot(x_positions, condition_data['tokens_same_pct_min'], 
                    marker='d', linewidth=2, markersize=8, 
                    label=label, color=color)

axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('Tokens Same / Min Length')
axes[1, 1].set_title('Tokens Same as % of Min Length (Conditioned)')
axes[1, 1].set_xticks(x_positions)
axes[1, 1].set_xticklabels(models, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tokens_same_conditioned.pdf")
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
for _, row in metrics_df.iterrows():
    print(f"{row['model']}:")
    print(f"  Vowel & Consonant Correct: {row['vowel_consonant_correct']:.3f}")
    print(f"  Vowel & Singular Consonant Correct: {row['vowel_singular_consonant_correct']:.3f}")
    print(f"  Vowel Correct: {row['vowel_correct']:.3f}")
    print(f"  Consonant Correct: {row['consonant_correct']:.3f}")
    print(f"  Singular Consonant Correct: {row['singular_consonant_correct']:.3f}")
    print(f"  Mean Tokens Same: {row['mean_tokens_same']:.3f}")
    print(f"  Tokens Same % of Original: {row['tokens_same_pct_orig']:.3f}")
    print(f"  Tokens Same % of Intervention: {row['tokens_same_pct_intervened']:.3f}")
    print(f"  Tokens Same % of Min Length: {row['tokens_same_pct_min']:.3f}")
    print()

# Print conditioned summary statistics
print("\nConditioned Summary Statistics:")
print("===============================")
for model in models:
    print(f"{model}:")
    model_conditioned = conditioned_df[conditioned_df['model'] == model]
    
    for _, row in model_conditioned.iterrows():
        condition = row['vowel_consonant_correct']
        print(f"  Vowel & Consonant Correct = {condition} (n={row['count']}):")
        print(f"    Mean Tokens Same: {row['mean_tokens_same']:.3f}")
        print(f"    Tokens Same % of Original: {row['tokens_same_pct_orig']:.3f}")
        print(f"    Tokens Same % of Intervention: {row['tokens_same_pct_intervened']:.3f}")
        print(f"    Tokens Same % of Min Length: {row['tokens_same_pct_min']:.3f}")
    print()
#%%
# Plot mean generation lengths by model
plt.figure(figsize=(12, 6))
x_positions = range(len(models))

# Calculate mean lengths for each model
mean_orig_lengths = []
mean_intervened_lengths = []
mean_min_lengths = []

for model in models:
    model_data = combined_df[combined_df['model'] == model]
    mean_orig_lengths.append(model_data['orig_len'].mean())
    mean_intervened_lengths.append(model_data['intervened_len'].mean())
    min_len = model_data[['orig_len', 'intervened_len']].min(axis=1)
    mean_min_lengths.append(min_len.mean())

plt.plot(x_positions, mean_orig_lengths, 
         marker='o', linewidth=2, markersize=8, 
         label='Original Generation Length', color='blue')

plt.plot(x_positions, mean_intervened_lengths, 
         marker='s', linewidth=2, markersize=8, 
         label='Intervention Generation Length', color='red')

plt.plot(x_positions, mean_min_lengths, 
         marker='^', linewidth=2, markersize=8, 
         label='Min Generation Length', color='green')

plt.xlabel('Model')
plt.ylabel('Mean Token Length')
plt.title('Mean Generation Length by Model')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mean_generation_lengths.pdf")
plt.show()

# Print length statistics
print("\nGeneration Length Statistics:")
print("=============================")
for i, model in enumerate(models):
    print(f"{model}:")
    print(f"  Mean Original Length: {mean_orig_lengths[i]:.2f}")
    print(f"  Mean Intervention Length: {mean_intervened_lengths[i]:.2f}")
    print(f"  Mean Min Length: {mean_min_lengths[i]:.2f}")
    print()
         
# %%
