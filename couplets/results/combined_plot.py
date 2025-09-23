#%%
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.corpus import cmudict

# Directories for unsteered and steered data
UNSTEERED_DIR = "couplets"
STEERED_DIR = "rhyme_intervention_sample"

models = ['Qwen3-0.6B', 'Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B']

# Initialize phonemizer for vowel analysis
phonemizer = cmudict.dict()

def get_last_vowel(word):
    """Extract the last vowel phoneme from a word."""
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

# Dictionary to store all metrics
results = {
    'model': [],
    'unsteered_rhyme': [],
    'unsteered_vowel': [],  # We'll need to calculate this from unsteered data
    'steered_rhyme': [],
    'steered_vowel': []
}

#%%
# Process each model
for model in models:
    print(f"Processing {model}...")

    # Load unsteered data
    unsteered_path = os.path.join(UNSTEERED_DIR, f"{model}.csv")
    unsteered_df = pd.read_csv(unsteered_path)
    
    # Calculate rhyme and vowel accuracy using the same logic as plot_results.py
    unsteered_df['vowel_match'] = None
    unsteered_df['consonant_match'] = None
    
    for idx, rhyme_group, second_last_word in zip(unsteered_df.index, unsteered_df['rhyme_group'], unsteered_df['second_last_word']):
        if pd.isna(second_last_word):
            unsteered_df.loc[idx, 'vowel_match'] = False
            unsteered_df.loc[idx, 'consonant_match'] = False
            continue
        unsteered_df.loc[idx, 'vowel_match'] = (get_last_vowel(rhyme_group) == get_last_vowel(second_last_word))
        unsteered_df.loc[idx, 'consonant_match'] = (get_final_consonant(rhyme_group) == get_final_consonant(second_last_word))

    # Perfect rhyme requires both vowel and consonant match
    unsteered_df['perfect_rhyme'] = unsteered_df['vowel_match'] & unsteered_df['consonant_match']
    
    unsteered_rhyme_rate = unsteered_df['perfect_rhyme'].mean()
    unsteered_vowel_rate = unsteered_df['vowel_match'].mean()
    
    # Load steered data  
    steered_path = os.path.join(STEERED_DIR, f"{model}.csv")
    steered_df = pd.read_csv(steered_path)
    
    # Calculate steered accuracy rates using the same logic as unsteered
    steered_df['vowel_match'] = None
    steered_df['consonant_match'] = None
    
    for idx, chosen_rhyme_group, intervention_generation in zip(steered_df.index, steered_df['chosen_rhyme_group'], steered_df['intervention_generation']):
        if pd.isna(intervention_generation):
            steered_df.loc[idx, 'vowel_match'] = False
            steered_df.loc[idx, 'consonant_match'] = False
            continue
        
        # Extract last word from intervention generation
        last_word = intervention_generation.split()[-1] if intervention_generation.split() else ""
        last_word = re.sub(r'^[^\w]+|[^\w]+$', '', last_word.strip()).lower()
        
        steered_df.loc[idx, 'vowel_match'] = (get_last_vowel(chosen_rhyme_group) == get_last_vowel(last_word))
        steered_df.loc[idx, 'consonant_match'] = (get_final_consonant(chosen_rhyme_group) == get_final_consonant(last_word))

    # Perfect rhyme requires both vowel and consonant match
    steered_df['perfect_rhyme'] = steered_df['vowel_match'] & steered_df['consonant_match']
    
    steered_rhyme_rate = steered_df['perfect_rhyme'].mean()
    steered_vowel_rate = steered_df['vowel_match'].mean()
    
    # Store results
    results['model'].append(model)
    results['unsteered_rhyme'].append(unsteered_rhyme_rate)
    results['unsteered_vowel'].append(unsteered_vowel_rate)
    results['steered_rhyme'].append(steered_rhyme_rate)
    results['steered_vowel'].append(steered_vowel_rate)

#%%
# Create the comparison plot
plt.figure(figsize=(12, 6))

x_positions = np.arange(len(models))

# Plot all four lines on the same plot - group blue lines first for legend ordering
plt.plot(x_positions, results['unsteered_rhyme'], 
         marker='o', linewidth=2, markersize=8, 
         label='Rhyming Accuracy', color='steelblue', linestyle='-')

plt.plot(x_positions, results['unsteered_vowel'], 
         marker='s', linewidth=2, markersize=8, 
         label='Assonant Rhyme Accuracy', color='steelblue', linestyle='--')

plt.plot(x_positions, results['steered_rhyme'], 
         marker='o', linewidth=2, markersize=8, 
         label='Steered Rhyming Accuracy', color='coral', linestyle='-')

plt.plot(x_positions, results['steered_vowel'], 
         marker='s', linewidth=2, markersize=8, 
         label='Steered Assonant Rhyme Accuracy', color='coral', linestyle='--')

#plt.xlabel('Model', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Rhyme and Assonant Rhyme Accuracy', fontsize=20)
plt.xticks(x_positions, models, rotation=0, ha='center', fontsize=20)
plt.legend(fontsize=17)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("combined_rhyme_accuracy.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%