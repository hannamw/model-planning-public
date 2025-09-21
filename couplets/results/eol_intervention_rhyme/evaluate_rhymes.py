#%%
from functools import lru_cache
import requests
import re

import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
import nltk
nltk.download('cmudict')
#%%
@lru_cache(maxsize=10000)
def fetch_rhymes(word: str) -> frozenset[str]:
    """Return a set of rhyming words for *word* (cached)."""
    word = word.lower()
    try:
        resp = requests.get("https://api.datamuse.com/words", params={"rel_rhy": word, "max": 1000}, timeout=20)
        resp.raise_for_status()
        return frozenset(item["word"].lower() for item in resp.json())
    except Exception as exc:  # noqa: BLE001  (broad ≈ network)
        print(f"Datamuse query failed for '{word}': {exc}")
        return frozenset()
#%%
def strip_s(s):
    return s[:-1] if s[-1] == 's' else s

phonemizer = cmudict.dict()

# Extract just vowels (phonemes ending in 0, 1, or 2)
def get_last_vowel(word):
    if word.lower() in phonemizer:
        phonemes = phonemizer[word.lower()][0]  # First pronunciation
        vowels = [p for p in phonemes if p[-1] in '012']
        return vowels[-1] if vowels else None
    return None

def get_final_consonant(word, depluralize=False):
    if word.lower() in phonemizer:
        phonemes = phonemizer[word.lower()][0]  # First pronunciation
        if not phonemes:
            return None
        if depluralize and phonemes[-1] == 'Z':
            phonemes = phonemes[:-1]
        if not phonemes:
            return None
        last_phoneme = phonemes[-1]
        return last_phoneme if last_phoneme[-1] not in '012' else None
    return None
#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

# Keep a list of all model dataframes with metrics
all_model_dfs = []
for model in models:
    print(model)
    df = pd.read_csv(model + '.csv')
    
    # Add columns for correctness metrics and vowel/consonant detection
    df['model'] = model
    df['rhyme_correct'] = False
    df['vowel_correct'] = False
    df['consonant_correct'] = False
    df['detected_vowel'] = None
    df['detected_consonant'] = None
    df['target_vowel'] = None
    df['target_consonant'] = None
    df['singular_consonant_correct'] = None
    
    for idx, row in df.iterrows():
        target_rhyme_group = row['rhyme_group']
        intervention_generation = row['stopped_generation']
        last_word = intervention_generation.split()[-1]
        last_word = re.sub(r'^[^\w]+|[^\w]+$', '', last_word.strip()).lower()
        
        # Check rhyme correctness
        rhyming_words = set() #fetch_rhymes(target_rhyme_group)
        df.loc[idx, 'rhyme_correct'] = last_word in rhyming_words

        # Get and compare vowels
        rhyme_vowel = get_last_vowel(target_rhyme_group)
        intervention_vowel = get_last_vowel(last_word)
        df.loc[idx, 'target_vowel'] = rhyme_vowel
        df.loc[idx, 'detected_vowel'] = intervention_vowel
        df.loc[idx, 'vowel_correct'] = rhyme_vowel == intervention_vowel

        # Get and compare consonants
        rhyme_consonant = get_final_consonant(target_rhyme_group)
        intervention_consonant = get_final_consonant(last_word)
        df.loc[idx, 'target_consonant'] = rhyme_consonant
        df.loc[idx, 'detected_consonant'] = intervention_consonant
        df.loc[idx, 'consonant_correct'] = rhyme_consonant == intervention_consonant

        rhyme_consonant_singular = get_final_consonant(target_rhyme_group, depluralize=True)
        intervention_consonant_singular = get_final_consonant(last_word, depluralize=True)
        df.loc[idx, 'singular_consonant_correct'] = rhyme_consonant_singular == intervention_consonant_singular

        print("Generated:", last_word)
        print("Original rhyme group:", target_rhyme_group)
        print("Rhyme correct:", df.loc[idx, 'rhyme_correct'])
        print("Vowel correct:", df.loc[idx, 'vowel_correct'])
        print("Consonant correct:", df.loc[idx, 'consonant_correct'])
        print("Singular Consonant correct:", df.loc[idx, 'singular_consonant_correct'])
        print("Target vowel:", df.loc[idx, 'target_vowel'])
        print("Detected vowel:", df.loc[idx, 'detected_vowel'])
        print("Target consonant:", df.loc[idx, 'target_consonant'])
        print("Detected consonant:", df.loc[idx, 'detected_consonant'])
        print("--------------------------------")
    
    # Save the updated dataframe with new columns
    df.to_csv(model + '.csv', index=False)
    print(f"Saved updated {model}.csv with computed columns")
    
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
    
    model_metrics.append({
        'model': model,
        'vowel_consonant_correct': vowel_consonant_correct,
        'vowel_singular_consonant_correct': vowel_singular_consonant_correct,
        'vowel_correct': vowel_correct,
        'consonant_correct': consonant_correct,
        'singular_consonant_correct': singular_consonant_correct
    })

metrics_df = pd.DataFrame(model_metrics)

# Plot the results
plt.figure(figsize=(12, 6))
x_positions = range(len(models))

plt.plot(x_positions, metrics_df['vowel_consonant_correct'], 
         marker='o', linewidth=2, markersize=8, 
         label='Vowel & Consonant Correct', color='blue')

plt.plot(x_positions, metrics_df['vowel_singular_consonant_correct'], 
         marker='s', linewidth=2, markersize=8, 
         label='Vowel & Singular Consonant Correct', color='red')

plt.plot(x_positions, metrics_df['vowel_correct'], 
         marker='^', linewidth=2, markersize=8, 
         label='Vowel Correct', color='green')

plt.plot(x_positions, metrics_df['consonant_correct'], 
         marker='d', linewidth=2, markersize=8, 
         label='Consonant Correct', color='orange')

plt.plot(x_positions, metrics_df['singular_consonant_correct'], 
         marker='v', linewidth=2, markersize=8, 
         label='Singular Consonant Correct', color='purple')

plt.xlabel('Model')
plt.ylabel('Accuracy Rate')
plt.title('Vowel and Consonant Correctness by Model')
plt.xticks(x_positions, models, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rhyme_steering.pdf")
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
    print()
         
# %%
