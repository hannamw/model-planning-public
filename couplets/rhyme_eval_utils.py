#%%
from functools import lru_cache
import requests
import re

import pandas as pd
from nltk.corpus import cmudict
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

def eval_df(orig_df: pd.DataFrame):
    df = orig_df.copy()
    df['rhyme_correct'] = False
    df['vowel_correct'] = False
    df['consonant_correct'] = False
    df['detected_vowel'] = None
    df['detected_consonant'] = None
    df['target_vowel'] = None
    df['target_consonant'] = None
    df['singular_consonant_correct'] = None
    
    for idx, row in df.iterrows():
        original_rhyme_group = row['rhyme_group']
        target_rhyme_group = row['chosen_rhyme_group']
        intervention_generation = row['intervention_generation']
        last_word = intervention_generation.split()[-1]
        last_word = re.sub(r'^[^\w]+|[^\w]+$', '', last_word.strip()).lower()
        
        # Check rhyme correctness
        rhyming_words = fetch_rhymes(target_rhyme_group)
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

    vowel_accuracy = df['vowel_correct']
    consonant_accuracy = df['consonant_correct']
    singular_consonant_accuracy = df['singular_consonant_correct']
    vowel_consonant_accuracy = df['vowel_correct'] & df['consonant_correct']
    vowel_singular_consonant_accuracy = df['vowel_correct'] & df['singular_consonant_correct']
    print(f"Vowel accuracy: {vowel_accuracy.mean()}")
    print(f"Consonant accuracy: {consonant_accuracy.mean()}")
    print(f"Singular consonant accuracy: {singular_consonant_accuracy.mean()}")
    print(f"Vowel and consonant accuracy: {vowel_consonant_accuracy.mean()}")
    print(f"Vowel and singular consonant accuracy: {vowel_singular_consonant_accuracy.mean()}")
    return df