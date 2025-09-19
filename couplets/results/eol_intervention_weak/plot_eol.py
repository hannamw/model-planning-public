#%%
import numpy as np
import pandas as pd

from transformers import AutoTokenizer
#%%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]
for model in models:
    df = pd.read_csv(f'{model}.csv')
    tokenizer = AutoTokenizer.from_pretrained(f'Qwen/{model}')

    print(model)
    print(f"Mean # features: ", df['n_features'].mean())
    print(f"Mean # selected features: ", df['n_selected_features'].mean())

    # fragment length vs. intervened length
    stopped_len_diffs = []
    for subs, stopped, orig in zip(df['substring'], df['stopped_generation'], df['original_generation']):
        subs = subs.replace('<|im_start|>', '').replace('<|im_end|>', '')
        stopped_clipped = stopped[len(subs):]
        orig_clipped = orig[len(subs):]

        stopped_toks = [tokenizer.decode(tok) for tok in tokenizer(stopped_clipped).input_ids]
        orig_toks = [tokenizer.decode(tok) for tok in tokenizer(orig_clipped).input_ids]

        for stopped_len, stopped_tok in enumerate(stopped_toks):
            if '\n' in stopped_tok:
                break
        
        for orig_len, orig_tok in enumerate(orig_toks):
            if '\n' in orig_tok:
                break
        stopped_len_diffs.append(stopped_len - orig_len)
    
    stopped_len_diffs = np.array(stopped_len_diffs)
    # normal length vs. continued length

    continued_len_diffs = []
    for continued, orig in zip(df['continued_generation'], df['original_generation']):
        continued_toks = [tokenizer.decode(tok) for tok in tokenizer(continued).input_ids]
        orig_toks = [tokenizer.decode(tok) for tok in tokenizer(orig).input_ids]
        continued_len_diffs.append(len(continued_toks) - len(orig_toks))
    continued_len_diffs = np.array(continued_len_diffs)

    print(f"Mean stopped length diff: {stopped_len_diffs.mean()}")
    print(f"Mean continued length diff: {continued_len_diffs.mean()}")
    
    # Add the numpy arrays as new columns to the dataframe
    df['eol_stopped_len_diffs'] = stopped_len_diffs
    df['eol_continued_len_diffs'] = continued_len_diffs
    
    # Save the updated dataframe
    df.to_csv(f'{model}.csv', index=False)
    print(f"Saved updated {model}.csv with new columns")
# %%
