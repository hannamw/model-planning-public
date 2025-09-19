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
    output_file = f"Qwen3-{size}_filtered.csv"
    
    print(f"Processing {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Initial rows: {len(df)}")
    
    # Strip 'user\n' from generation column
    df['generation'] = df['generation'].apply(strip_user_newline)
    
    # Remove first len(input_text) characters from both generation columns
    char_results = df.apply(lambda row: remove_input_text_chars(row['baseline_generation'], row['generation'], row['input_text']), axis=1)
    df['baseline_generation'] = [result[0] for result in char_results]
    df['generation'] = [result[1] for result in char_results]
    
    # Filter on contains_steered_word being True
    df = df[df['contains_steered_word'] == True]
    print(f"After filtering contains_steered_word=True: {len(df)}")
    
    # Filter on is_coherent being True
    df = df[df['is_coherent'] == True]
    print(f"After filtering is_coherent=True: {len(df)}")
    
    # Find first differing word and check if word_steered is prefix
    df['first_diff_word'] = df.apply(lambda row: find_first_differing_word(row['baseline_generation'], row['generation']), axis=1)
    df['steered_is_prefix'] = df.apply(lambda row: is_prefix(row['word_steered'], row['first_diff_word']), axis=1)
    
    # Filter out examples where word_steered is prefix of first differing word
    df_filtered = df[df['steered_is_prefix'] == False]
    print(f"After filtering out steered_is_prefix=True: {len(df_filtered)}")
    
    # Drop helper columns
    df_filtered = df_filtered.drop(columns=['first_diff_word', 'steered_is_prefix'])
    
    # Save filtered dataframe
    df_filtered.to_csv(output_file, index=False)
    print(f"Saved to {output_file}\n")

# %%
