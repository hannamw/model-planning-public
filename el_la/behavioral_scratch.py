#%%
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained_no_processing('Qwen/Qwen3-8B', dtype=torch.bfloat16)
# %%
ex = 'El roedor que vive cerca del agua es la rata almizclera. La tela que cubre el colchón donde duermes es la'
generation = model.generate(ex, do_sample=False)
print(generation)
# %%
