#%%
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import cmudict

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

# Folder that holds all model-specific CSVs

rhyme_accuracies = {}
vowel_accuracies = {}

for model_name in [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14, 32]]:
    csv_path = f'{model_name}.csv'
    df = pd.read_csv(csv_path)
    df['vowel_match'] = None
    df['consonant_math'] = None
    for idx, rhyme_group, second_last_word in zip(df.index, df['rhyme_group'], df['second_last_word']):
        if pd.isna(second_last_word):
            df.loc[idx, 'vowel_match'] = False
            df.loc[idx, 'consonant_match'] = False
            continue
        df.loc[idx, 'vowel_match'] = (get_last_vowel(rhyme_group) == get_last_vowel(second_last_word))
        df.loc[idx, 'consonant_match'] = (get_final_consonant(rhyme_group) == get_final_consonant(second_last_word))

    df['rhyme_success'] = df['vowel_match'] & df['consonant_match']

    rhyme_accuracies[model_name] = df['rhyme_success'].mean()
    vowel_accuracies[model_name] = df['vowel_match'].mean()

# --- Plotting ---------------------------------------------------------------
models = list(rhyme_accuracies.keys())
rhyme_scores = [rhyme_accuracies[m] for m in models]
vowel_scores = [vowel_accuracies[m] for m in models]

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 14})

# Create line plots
plt.plot(models, rhyme_scores, marker='o', linewidth=2.5, markersize=8, 
         label='Rhyme Accuracy (Perfect Rhyme)', color='steelblue')
plt.plot(models, vowel_scores, marker='s', linewidth=2.5, markersize=8, 
         label='Vowel Match Accuracy (Assonant Rhyme)', color='darkred')

plt.ylabel("Accuracy", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.title("Rhyming Accuracy by Model", fontsize=18)
plt.ylim(0, 1.05)
plt.xticks(rotation=0, ha="center", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rhyming_accuracy_line.pdf")
plt.show()
# %%
