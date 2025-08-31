#%%
import torch
from circuit_tracer import ReplacementModel 
import json
#%%
models_to_configs = {
    'Qwen/Qwen3-0.6B': 'qwen3-0.6b-relu-lowl0',
    'Qwen/Qwen3-1.7B': 'qwen3-1.7b-relu-lowl0',
    'Qwen/Qwen3-4B': 'qwen3-4b-relu',
    'Qwen/Qwen3-8B': 'qwen3-8b-relu',
    'Qwen/Qwen3-14B': 'qwen3-14b-relu-lowl0',
}

models_and_transcoders = {
    'Qwen/Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen/Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen/Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen/Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen/Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"}

def compute_batched_effects(W_dec, W_U, batch_size=1000):
    """Compute W_dec @ W_U in batches to avoid memory issues"""
    d_feat, d_model = W_dec.shape
    d_vocab = W_U.shape[1]
    
    effects = torch.zeros(d_feat, d_vocab, dtype=W_dec.dtype, device=W_dec.device)
    
    for start_idx in range(0, d_feat, batch_size):
        end_idx = min(start_idx + batch_size, d_feat)
        batch_effects = W_dec[start_idx:end_idx] @ W_U
        effects[start_idx:end_idx] = batch_effects
    
    return effects

def get_feature_top_bottom_logits(effects, tokenizer, top_k=10):
    """For each feature, get the top and bottom k tokens by effect value"""
    d_feat, d_vocab = effects.shape
    feature_logits = {}
    
    for feat_idx in range(d_feat):
        feature_effects = effects[feat_idx]  # effects for this feature across all tokens
        
        # Get top k tokens with highest effects
        top_values, top_indices = torch.topk(feature_effects, top_k)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
        
        # Get bottom k tokens with lowest effects
        bottom_values, bottom_indices = torch.topk(feature_effects, top_k, largest=False)
        bottom_tokens = [tokenizer.decode([idx.item()]) for idx in bottom_indices]
        
        feature_logits[feat_idx] = {
            "top_logits": top_tokens,
            "bottom_logits": bottom_tokens
        }
    
    return feature_logits

for model_name, config in models_to_configs.items():
    print(f"Processing {model_name}...")
    model_name_noslash = model_name.split('/')[-1]
    transcoders = models_and_transcoders[model_name]
    model = ReplacementModel.from_pretrained(model_name, transcoders, dtype=torch.bfloat16, lazy_encoder=True)
    
    W_U = model.W_U  # d_model, d_vocab
    tokenizer = model.tokenizer
    
    for i in range(model.cfg.n_layers):
        print(f"  Layer {i}/{model.cfg.n_layers-1}")
        W_dec = model.transcoders.transcoders[i].W_dec  # d_feat, d_model
        
        # Compute effects with batching
        effects = compute_batched_effects(W_dec, W_U, batch_size=1000)
        
        # Get top and bottom logits for each feature
        feature_logits = get_feature_top_bottom_logits(effects, tokenizer)
        
        # Save results
        results = {
            "model_name": model_name,
            "layer": i,
            "feature_logits": feature_logits
        }
        
        filename = f"../cache/top_logits/{model_name_noslash}-layer{i}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Clear GPU memory
        del effects
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clear model from memory
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
# %%
