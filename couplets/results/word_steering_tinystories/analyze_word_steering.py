#%%
import pandas as pd
from collections import defaultdict

#%%
models = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']

# Function to extract parameter count for sorting
def get_param_count(model_name):
    if 'Qwen3-0.6B' in model_name:
        return 0.6
    elif 'Qwen3-1.7B' in model_name:
        return 1.7
    elif 'Qwen3-4B' in model_name:
        return 4.0
    elif 'Qwen3-8B' in model_name:
        return 8.0
    elif 'Qwen3-14B' in model_name:
        return 14.0
    else:
        return 0.0

all_results = []

for model in models:
    print(f"Processing {model}...")
    
    df = pd.read_csv(f'{model}.csv')
    
    # Filter out rows with empty word_steered (baseline generations) and "user" separators
    steered_df = df[
        (df['word_steered'].notna()) & 
        (df['word_steered'] != '') & 
        (df['generation'] != 'user')
    ].copy()
    
    # Group by sentence_index and word_steered to analyze each steering attempt
    grouped = steered_df.groupby(['sentence_index', 'word_steered'])
    
    word_success_rates = {}
    word_feature_counts = defaultdict(list)
    
    for (sentence_idx, word_steered), group in grouped:
        # Strip input_text from each generation before checking
        input_text = group.iloc[0]['input_text']
        
        # Check if any generation contains the steered word (after stripping input)
        contains_word = False
        for _, row in group.iterrows():
            generation = row['generation']
            # Strip the input_text from the beginning of generation
            if generation.startswith(input_text):
                generation_only = generation[len(input_text):].strip()
            else:
                generation_only = generation
            
            # Check if the steered word appears in the generation (case insensitive)
            if word_steered.lower() in generation_only.lower():
                contains_word = True
                break
        
        # Track success for this word
        if word_steered not in word_success_rates:
            word_success_rates[word_steered] = []
        word_success_rates[word_steered].append(contains_word)
        
        # Collect feature counts for this word
        for _, row in group.iterrows():
            if pd.notna(row['feature_count']):
                word_feature_counts[word_steered].append(row['feature_count'])
    
    # Calculate success rates and mean feature counts for each word
    for word_steered in word_success_rates:
        success_rate = sum(word_success_rates[word_steered]) / len(word_success_rates[word_steered])
        mean_feature_count = sum(word_feature_counts[word_steered]) / len(word_feature_counts[word_steered]) if word_feature_counts[word_steered] else 0
        
        all_results.append({
            'model': model,
            'word_steered': word_steered,
            'success_rate': success_rate,
            'mean_feature_count': mean_feature_count,
            'num_examples': len(word_success_rates[word_steered])
        })

# Create results DataFrame
results_df = pd.DataFrame(all_results)

# Add parameter count column for sorting
results_df['param_count'] = results_df['model'].apply(get_param_count)

# Sort by parameter count and word_steered
results_df = results_df.sort_values(['param_count', 'word_steered'])

# Drop the helper column
results_df = results_df.drop('param_count', axis=1)

print(f"\nSummary:")
print(results_df.to_string(index=False))

# Print summary statistics
print(f"\nOverall Statistics:")
print(f"Total unique words steered: {results_df['word_steered'].nunique()}")
print(f"Models analyzed: {', '.join(results_df['model'].unique())}")

# Average success rate by model (sorted by parameter count)
model_avg = results_df.groupby('model')['success_rate'].mean()
print(f"\nAverage success rate by model:")
for model in models:  # Use original models list to maintain order
    if model in model_avg:
        print(f"  {model}: {model_avg[model]:.3f}")

# Average feature count by model (sorted by parameter count)
model_feature_avg = results_df.groupby('model')['mean_feature_count'].mean()
print(f"\nAverage feature count by model:")
for model in models:  # Use original models list to maintain order
    if model in model_feature_avg:
        print(f"  {model}: {model_feature_avg[model]:.2f}")

# %%
