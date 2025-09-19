#%%
import pandas as pd

def strip_user_newline(text):
    if isinstance(text, str) and text.startswith('user\n'):
        return text[5:]
    return text

def word_in_generation(generation, word_steered):
    if pd.isna(generation) or pd.isna(word_steered):
        return False
    return str(word_steered).lower() in str(generation).lower()

def find_first_differing_word(baseline, generation):
    if pd.isna(baseline) or pd.isna(generation):
        return None
    
    baseline_words = str(baseline).split()
    generation_words = str(generation).split()
    
    min_len = min(len(baseline_words), len(generation_words))
    
    for i in range(min_len):
        if baseline_words[i] != generation_words[i]:
            return generation_words[i]
    
    if len(generation_words) > len(baseline_words):
        return generation_words[min_len]
    
    return None

def remove_input_text_chars(baseline, generation, input_text):
    """Remove the first len(input_text) characters from both baseline_generation and generation."""
    if pd.isna(baseline) or pd.isna(generation) or pd.isna(input_text):
        return baseline, generation
    
    num_chars_to_remove = len(str(input_text))
    
    baseline_str = str(baseline)
    generation_str = str(generation)
    
    # Remove the first num_chars_to_remove characters
    baseline_trimmed = baseline_str[num_chars_to_remove:] if len(baseline_str) > num_chars_to_remove else ""
    generation_trimmed = generation_str[num_chars_to_remove:] if len(generation_str) > num_chars_to_remove else ""
    
    return baseline_trimmed, generation_trimmed

def is_prefix(word_steered, first_diff_word):
    if pd.isna(word_steered) or first_diff_word is None:
        return False
    return str(first_diff_word).lower().startswith(str(word_steered).lower())

# Process CSV files for Qwen3 models (0.6B, 1.7B, 4B, 8B, 14B)

model_sizes = ["0.6B", "1.7B", "4B", "8B", "14B"]

for size in model_sizes:
    input_file = f"Qwen3-{size}_analyzed.csv"
    annotated_file = f"Qwen3-{size}_filtered_annotated.csv"
    
    print(f"Processing {input_file}")
    
    df = pd.read_csv(input_file)
    df_annotated = pd.read_csv(annotated_file, delimiter=';')
    df_annotated = df_annotated[~pd.isna(df_annotated['adapted_context_manual'])]
    print(f"Initial rows: {len(df_annotated)}")
    
    # Strip 'user\n' from generation column
    df['generation'] = df['generation'].apply(strip_user_newline)
    
    # Remove first len(input_text) characters from both generation columns
    char_results = df.apply(lambda row: remove_input_text_chars(row['baseline_generation'], row['generation'], row['input_text']), axis=1)
    df['baseline_generation'] = [result[0] for result in char_results]
    df['generation'] = [result[1] for result in char_results]
    
    # Find first differing word and check if word_steered is prefix
    df['first_diff_word'] = df.apply(lambda row: find_first_differing_word(row['baseline_generation'], row['generation']), axis=1)
    df['steered_is_prefix'] = df.apply(lambda row: is_prefix(row['word_steered'], row['first_diff_word']), axis=1)

    un_acc = (df['contains_steered_word'] & df['is_coherent'] & ~df['steered_is_prefix']).mean()
    print(f"Overall metrics for {size}:")
    print(f"  Unqualified accuracy: {un_acc:.4f}")
    print(f"  Manual adaptation rate: {df_annotated['adapted_context_manual'].mean():.4f}")
    print(f"  Combined metric: {un_acc * df_annotated['adapted_context_manual'].mean():.4f}")
    
    # Compute metrics per steering strength
    print(f"\nMetrics per steering strength for {size}:")
    steering_strengths = sorted(df['steering_strength'].unique())
    
    for strength in steering_strengths:
        print(strength)
        df_strength = df[df['steering_strength'] == strength]
        
        if len(df_strength) > 0:
            un_acc_strength = (df_strength['contains_steered_word'] & 
                             df_strength['is_coherent'] & 
                             ~df_strength['steered_is_prefix']).mean()
            
            # Get corresponding annotated data for this steering strength
            # We need to match by some identifier - let's check if we can match by index or other columns
            df_annotated_strength = df_annotated[df_annotated.index.isin(df_strength.index)]
            
            if len(df_annotated_strength) > 0:
                manual_adaptation_rate = df_annotated_strength['adapted_context_manual'].mean()
                combined_metric = un_acc_strength * manual_adaptation_rate
                
                print(df_strength['contains_steered_word'].mean())
                print(df_strength['is_coherent'].mean())
                print(1 - df_strength['steered_is_prefix'].mean())
                print(manual_adaptation_rate)
                print(combined_metric)
    
    print("-" * 60)
    
#%%
# Create bar plot with all metrics per model
import matplotlib.pyplot as plt
import numpy as np

# Collect metrics for all models
model_sizes = ["0.6B", "1.7B", "4B", "8B", "14B"]
metrics_data = {
    'contains_word': [],
    'is_coherent': [],
    'adapted_context': [],
    'combined': []
}

for size in model_sizes:
    input_file = f"Qwen3-{size}_analyzed.csv"
    annotated_file = f"Qwen3-{size}_filtered_annotated.csv"
    
    df = pd.read_csv(input_file)
    df_annotated = pd.read_csv(annotated_file, delimiter=';')
    df_annotated = df_annotated[~pd.isna(df_annotated['adapted_context_manual'])]
    
    # Strip 'user\n' from generation column
    df['generation'] = df['generation'].apply(strip_user_newline)
    
    # Remove first len(input_text) characters from both generation columns
    char_results = df.apply(lambda row: remove_input_text_chars(row['baseline_generation'], row['generation'], row['input_text']), axis=1)
    df['baseline_generation'] = [result[0] for result in char_results]
    df['generation'] = [result[1] for result in char_results]
    
    # Find first differing word and check if word_steered is prefix
    df['first_diff_word'] = df.apply(lambda row: find_first_differing_word(row['baseline_generation'], row['generation']), axis=1)
    df['steered_is_prefix'] = df.apply(lambda row: is_prefix(row['word_steered'], row['first_diff_word']), axis=1)
    
    # Compute metrics
    contains_word = df['contains_steered_word'].mean()
    is_coherent = df['is_coherent'].mean()
    adapted_context = df_annotated['adapted_context_manual'].mean()
    combined = (df['contains_steered_word'] & df['is_coherent'] & ~df['steered_is_prefix']).mean() * adapted_context
    
    metrics_data['contains_word'].append(contains_word)
    metrics_data['is_coherent'].append(is_coherent)
    metrics_data['adapted_context'].append(adapted_context)
    metrics_data['combined'].append(combined)

# Create the bar plot
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(model_sizes))
width = 0.2

bars1 = ax.bar(x - 1.5*width, metrics_data['contains_word'], width, label='Contains Word', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, metrics_data['is_coherent'], width, label='Is Coherent', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, metrics_data['adapted_context'], width, label='Adapted Context', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, metrics_data['combined'], width, label='All Conditions', alpha=0.8)

ax.set_xlabel('Model Size')
ax.set_ylabel('Frequency')
ax.set_title('Model Steering Metrics by Size')
ax.set_xticks(x)
ax.set_xticklabels([f'Qwen3-{size}' for size in model_sizes])
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

plt.tight_layout()
plt.savefig('steering_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
