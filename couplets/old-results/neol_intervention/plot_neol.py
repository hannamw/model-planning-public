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

    # fragment length vs. intervened length
    stopped_len_diffs = []
    for first_last_word, subs, stopped, orig in zip(df['first_last_word'], df['substring'], df['stopped_generation'], df['original_generation']):
        subs = subs.replace('<|im_start|>', '').replace('<|im_end|>', '')
        stopped_clipped = stopped[len(subs):]
        orig_clipped = orig[len(subs):]

        rhyming_words = fetch_rhymes(first_last_word)

        for stopped_len, stopped_word in enumerate(stopped_clipped.split()):
            stopped_word = re.sub(r'[^a-zA-Z]', '', stopped_word)
            if stopped_word in rhyming_words:
                break
        
        for orig_len, orig_word in enumerate(orig_clipped.split()):
            orig_word = re.sub(r'[^a-zA-Z]', '', orig_word)

            if orig_word in rhyming_words:
                break
        stopped_len_diffs.append(stopped_len - orig_len)
    
    stopped_len_diffs = np.array(stopped_len_diffs)
    # normal length vs. continued length

    continued_len_diffs = []
    eventually_rhymed = []
    for first_last_word, orig_input, continued, orig in zip(df['first_last_word'], df['original_input'], df['continued_generation'], df['original_generation']):
        orig_input = orig_input.replace('<|im_start|>', '').replace('<|im_end|>', '')
        orig_clipped = orig[len(orig_input):]
        continued_clipped = continued[len(orig_input):]

        rhyming_words = fetch_rhymes(first_last_word)

        continued_rhymed = False
        for continued_len, continued_word in enumerate(continued_clipped.split()):
            continued_word = re.sub(r'[^a-zA-Z]', '', continued_word)
            if continued_word in rhyming_words:
                continued_rhymed = True
                break
        
        for orig_len, orig_word in enumerate(orig_clipped.split()):
            orig_word = re.sub(r'[^a-zA-Z]', '', orig_word)

            if orig_word in rhyming_words:
                break
        
        continued_len_diffs.append(continued_len - orig_len)
        eventually_rhymed.append(continued_rhymed)
    continued_len_diffs = np.array(continued_len_diffs)
    eventually_rhymed = np.array(eventually_rhymed)

    print(f"Mean stopped length diff: {stopped_len_diffs.mean()}")
    print(f"Mean continued length diff: {continued_len_diffs.mean()}")
    print(f"Proportion where a rhyme happened: {eventually_rhymed.mean()}")
# %%
