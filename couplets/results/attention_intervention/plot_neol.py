#%%
import re
import requests
from functools import lru_cache
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
#%%
DATAMUSE_URL = "https://api.datamuse.com/words"
@lru_cache
def fetch_rhymes(word: str) -> frozenset[str]:
    """Datamuse rhymes with caching."""
    word = word.lower()
    try:
        resp = requests.get(DATAMUSE_URL, params={"rel_rhy": word, "max": 1000}, timeout=20)
        resp.raise_for_status()
        return frozenset(item["word"].lower() for item in resp.json())
    except Exception:  # noqa: BLE001
        print(f"Datamuse query failed for '{word}'.")
        return frozenset()

#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]
for model in models:
    df = pd.read_csv(f'{model}.csv')
    tokenizer = AutoTokenizer.from_pretrained(f'Qwen/{model}')
    print(model)

    # normal length vs. continued length
    continued_len_diffs = []
    eventually_rhymed = []
    for first_last_word, orig_input, new, orig in zip(df['first_last_word'], df['substring'], df['ablated_generation'], df['original_generation']):
        orig_input = orig_input.replace('<|im_start|>', '').replace('<|im_end|>', '')
        orig_clipped = orig[len(orig_input):]
        new_clipped = new[len(orig_input):]

        rhyming_words = fetch_rhymes(first_last_word)

        new_rhymed = False
        for new_len, new_word in enumerate(new_clipped.split()):
            new_word = re.sub(r'[^a-zA-Z]', '', new_word)
            if new_word in rhyming_words:
                new_rhymed = True
                break
        
        for orig_len, orig_word in enumerate(orig_clipped.split()):
            orig_word = re.sub(r'[^a-zA-Z]', '', orig_word)

            if orig_word in rhyming_words:
                break
        
        continued_len_diffs.append(new_len - orig_len)
        eventually_rhymed.append(new_rhymed)
    continued_len_diffs = np.array(continued_len_diffs)
    eventually_rhymed = np.array(eventually_rhymed)


    print(f"Mean ablated length diff: {continued_len_diffs.mean()}")
    print(f"Proportion where a rhyme happened: {eventually_rhymed.mean()}")
    
    # Add the numpy arrays as new columns to the dataframe
    df['neol_ablated_len_diffs'] = continued_len_diffs
    df['neol_eventually_rhymed'] = eventually_rhymed
    
    # Save the updated dataframe
    df.to_csv(f'{model}.csv', index=False)
    print(f"Saved updated {model}.csv with new columns")
# %%
