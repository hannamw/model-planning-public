#%%
from transformer_lens import HookedTransformer
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import namedtuple

from utils import create_dataset_examples

Example = namedtuple("Example", ["sentence", "continuation", "name"])

def get_logit_lens_all_layers(model, cache, k=5):
    """Get logit lens results for all layers"""
    n_layers = model.cfg.n_layers
    
    # Get token IDs for "a" and "an"
    a_token_id = model.tokenizer.encode(" a", add_special_tokens=False)[0]
    an_token_id = model.tokenizer.encode(" an", add_special_tokens=False)[0]
    
    # Storage for results
    top_k_probs = []
    top_k_tokens = []
    a_an_probs = []
    
    for layer in range(n_layers):
        # Get residual stream at this layer
        residual = cache["resid_post", layer]
        
        # Apply final layernorm
        final_layernorm_out = model.ln_final(residual)
        
        # Apply unembedding (W_U matrix)
        logits = final_layernorm_out @ model.W_U
        
        # Get probabilities for the last token position
        probs = F.softmax(logits.squeeze()[-1], dim=-1)
        
        # Get top-k tokens and probabilities
        topk = torch.topk(probs, k)
        layer_top_k_probs = topk.values.cpu()
        layer_top_k_tokens = topk.indices.cpu()
        
        # Get specific probabilities for "a" and "an"
        a_prob = probs[a_token_id].item()
        an_prob = probs[an_token_id].item()
        
        top_k_probs.append(layer_top_k_probs)
        top_k_tokens.append(layer_top_k_tokens)
        a_an_probs.append([a_prob, an_prob])
    
    return torch.stack(top_k_probs), torch.stack(top_k_tokens), torch.tensor(a_an_probs)


# Model configurations (based on compute_a_an_graphs.py)
model_names = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-14B'
]

# Load dataset
df = pd.read_csv('data/professions_dataset_with_articles.csv')
df_ex = create_dataset_examples(df)

# Create output directory
output_dir = Path('results/logit_lens')
output_dir.mkdir(exist_ok=True)

k = 5  # Number of top tokens to store

for model_name in model_names:
    print(f"Processing {model_name}...")
    
    # Load model using HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16
    )
    
    # Create examples as in compute_a_an_graphs.py
    examples = [Example(f"{model.tokenizer.eos_token}{prompt}", f' {article}', f'{article}-{profession}') 
                for prompt, article, profession in zip(df_ex['Prompt'], df_ex['Article'], df_ex['Profession'])]
    
    n_examples = len(examples)
    n_layers = model.cfg.n_layers
    
    # Initialize storage tensors
    all_top_k_probs = torch.zeros((n_examples, n_layers, k))
    all_top_k_tokens = torch.zeros((n_examples, n_layers, k), dtype=torch.int32)
    all_a_an_probs = torch.zeros((n_examples, n_layers, 2))  # [a_prob, an_prob]
    
    # Store metadata
    sentences = []
    correct_articles = []
    professions = []
    
    for i, (sentence, continuation, name) in enumerate(examples):
        print(f"  Processing example {i+1}/{n_examples}: {name}")
        
        with torch.inference_mode():
            # Run model with cache to get residual stream at all layers
            logits, cache = model.run_with_cache(sentence)
            
            # Get logit lens results for all layers
            top_k_probs, top_k_tokens, a_an_probs = get_logit_lens_all_layers(model, cache, k=k)
            
            # Store results
            all_top_k_probs[i] = top_k_probs
            all_top_k_tokens[i] = top_k_tokens
            all_a_an_probs[i] = a_an_probs
            
            # Store metadata
            sentences.append(sentence)
            correct_articles.append(name.split('-')[0])
            professions.append(name.split('-')[1])
    
    # Create model-specific output directory
    model_short_name = model_name.split('/')[-1]
    model_output_dir = output_dir / model_short_name
    model_output_dir.mkdir(exist_ok=True)
    
    # Create token strings for easier interpretation
    token_strings = []
    for example_tokens in all_top_k_tokens:
        example_token_strings = []
        for layer_tokens in example_tokens:
            layer_token_strings = [model.tokenizer.decode([token_id]) for token_id in layer_tokens]
            example_token_strings.append(layer_token_strings)
        token_strings.append(example_token_strings)
    
    # Save results as torch tensors in dictionary format
    results_dict = {
        'top_k_probs': all_top_k_probs,
        'top_k_tokens': all_top_k_tokens,
        'a_an_probs': all_a_an_probs,
        'token_strings': token_strings,
    }
    
    torch.save(results_dict, model_output_dir / 'results.pt')
    
    # Save metadata as pandas DataFrame
    metadata_df = pd.DataFrame({
        'sentences': sentences,
        'correct_articles': correct_articles,
        'professions': professions,
    })
    
    metadata_df.to_csv(model_output_dir / 'metadata.csv', index=False)
    
    print(f"  Saved results for {model_name} to {model_output_dir}")
    print(f"  Shape: {n_examples} examples x {n_layers} layers x {k} top tokens")
    
    # Free memory
    del model
    torch.cuda.empty_cache()

print(f"\nAll results saved to {output_dir}")