#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from circuit_tracer import ReplacementModel
from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
# %%
model = ReplacementModel.from_pretrained('Qwen/Qwen3-14B',"mwhanna/qwen3-14b-transcoders-lowl0", 
        dtype=torch.bfloat16, cpu_encoder=False, lazy_encoder=True, lazy_decoder=True, move_to_device=True)
# %%
transcoder_set = load_transcoder_from_hub("mwhanna/qwen3-14b-transcoders-lowl0", cpu_encoder=True, lazy_decoder=True)
# %%
transcoder_set[0].transcoders[0].W_enc.device
# %%
W_enc = transcoder_set[0].transcoders[0].W_enc
# %%
W_enc.device
# %%
transcoder_set[0].transcoders[0].device
# %%
model.W_E.device
# %%
model.transcoders.transcoders[0].W_enc
# %%
logits, activations = model.get_activations("This is a test")
# %%
