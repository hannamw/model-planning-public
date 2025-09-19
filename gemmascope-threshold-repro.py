#%%
import torch
from circuit_tracer import ReplacementModel
# %%
model = ReplacementModel.from_pretrained('google/gemma-2-2b', "mntss/gemma-scope-transcoders")
# %%
cache, caching_hooks, _ = model.get_caching_hooks(
    lambda name: model.feature_input_hook in name
)
logits = model.run_with_hooks("This is a test", fwd_hooks=caching_hooks)

# %%
from huggingface_hub import hf_hub_download
import numpy as np

path_to_params = hf_hub_download(
    repo_id="google/gemma-scope-2b-pt-transcoders",
    filename="layer_0/width_16k/average_l0_76/params.npz",
    force_download=False,
)
#%%
params = np.load(path_to_params)
pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
#%%
import torch.nn as nn
class JumpReLUSAE(nn.Module):
  def __init__(self, d_model, d_sae):
    # Note that we initialise these to zeros because we're loading in pre-trained weights.
    # If you want to train your own SAEs then we recommend using blah
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def encode(self, input_acts):
    pre_acts = input_acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    acts = mask * torch.nn.functional.relu(pre_acts)
    return acts

  def decode(self, acts):
    return acts @ self.W_dec + self.b_dec

  def forward(self, acts):
    acts = self.encode(acts)
    recon = self.decode(acts)
    return recon
#%%
sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
sae.load_state_dict(pt_params)
sae.to('cuda')
#%%
x = cache['blocks.0.ln2.hook_normalized']
y_hat_ours = model.transcoders.transcoders[0](x)
#%%
y_hat_reference = sae(x)
# %%
print(sae.threshold)
# %%
print(model.transcoders.transcoders[0].activation_function.threshold)
# %%
model.transcoders.transcoders[0].activation_function.threshold = sae.threshold
# %%
y_hat_ours_fixed = model.transcoders.transcoders[0](x)
# %%
print("Reference transcoder prediction:")
print(y_hat_reference)
print("Our transcoder prediction:")
print(y_hat_ours)
print("Our transcoder prediction (fixed)")
print(y_hat_ours_fixed)
# %%
pt_params['threshold']
# %%
