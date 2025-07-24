#%%
import pandas as pd
from transformer_lens import HookedTransformer
#%%
prompt = """/no_think Here’s the first sentence of a story: {first_sentence} Continue this story with one sentence that introduces a new animal character."""
#%%
df = pd.read_csv('results/tinystories/Qwen3-14B_tinystories_results.csv')
#%%
model = HookedTransformer.from_pretrained("Qwen/Qwen3-14B")
# %%
df['generated_continuation'][0]
# %%
messages = [
    {'role': 'user', 
    'content': prompt.format(first_sentence=df['first_sentence'][0])},
    {'role': 'assistant',
    'content': "One"}
]
inputs = model.tokenizer.apply_chat_template(messages, tokenize=False)[:-11]
logits, cache = model.run_with_cache(inputs)
# %%
messages2 = [
    {'role': 'user', 
    'content': prompt.format(first_sentence=df['first_sentence'][1])},
    {'role': 'assistant',
    'content': "One"}
]
inputs2 = model.tokenizer.apply_chat_template(messages2, tokenize=False)[:-11]
def replacement_hook(activations, hook):
    activations[:, -1] = cache[hook.name][:, -1]
    return activations

def remove_all_hook(activations, hook):
    model.remove_all_hook_fns()

hooks = [(f'blocks.{layer}.hook_resid_pre', replacement_hook) for layer in range(model.cfg.n_layers)]
hooks += [(f'blocks.{layer}.hook_resid_post', replacement_hook) for layer in range(model.cfg.n_layers)]

hooks.append(('ln_final.hook_normalized', remove_all_hook))
# %%
with model.hooks(hooks):
    generation = model.generate(inputs2, max_new_tokens=30, do_sample=False)
print(generation)
# %%
