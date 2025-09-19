#%%
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_model_results(model_name: str, results_dir: str = 'results/logit-lens') -> Tuple:
    """Load results and metadata for a specific model"""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    
    # Load results tensor
    results_path = model_dir / 'results.pt'
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    results = torch.load(results_path, map_location='cpu')
    
    # Load metadata
    metadata_path = model_dir / 'metadata.csv'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path)

    # Add missing columns: correct?, p(a), p(an)
    columns_to_add = []
    
    # If the "correct?" column is missing, infer it from the last-layer top-1 predictions
    if 'correct?' not in metadata.columns:
        # `token_strings` is a list (len = n_examples) where each element is a list of tokens per layer
        token_strings_all = results['token_strings']

        correct_flags = []
        for ex_tokens, correct_article in zip(token_strings_all, metadata['correct_articles']):
            # Get the top-1 token from the final layer
            last_layer_tokens = ex_tokens[-1]
            # `last_layer_tokens` is expected to be a list with the top-k tokens; take the first entry
            predicted_token = str(last_layer_tokens[0]).strip()
            correct_flags.append(predicted_token == correct_article)

        metadata['correct?'] = correct_flags
        columns_to_add.append('correct?')
    
    # Add p(a) and p(an) columns if missing
    if 'p(a)' not in metadata.columns or 'p(an)' not in metadata.columns:
        a_an_probs = results['a_an_probs']
        # Get final layer probabilities: shape (n_examples, 2)
        final_layer_probs = a_an_probs[:, -1, :]
        
        if 'p(a)' not in metadata.columns:
            metadata['p(a)'] = final_layer_probs[:, 0].numpy()
            columns_to_add.append('p(a)')
        
        if 'p(an)' not in metadata.columns:
            metadata['p(an)'] = final_layer_probs[:, 1].numpy()
            columns_to_add.append('p(an)')
    
    # Persist the updated metadata if any columns were added
    if columns_to_add:
        metadata.to_csv(metadata_path, index=False)

    return results, metadata

def create_article_probability_heatmap(model_name: str, 
                                     article_filter: Optional[str] = None,
                                     correct_filter: Optional[bool] = None,
                                     figsize: Tuple[int, int] = (8, 12),
                                     results_dir: str = 'results/logit-lens',
                                     ax: Optional[plt.Axes] = None):
    """
    Create a vertical heatmap of average probability of correct article by layer
    
    Args:
        model_name: Name of the model to analyze
        article_filter: Filter examples by article ('a', 'an', or None for all)
        correct_filter: Filter for correct (True) or incorrect (False) predictions.
        figsize: Figure size tuple (only used if ax is None)
        results_dir: Directory containing the results
        ax: Optional matplotlib axes to plot on
        
    Returns:
        tuple: (fig, ax, avg_probs) where fig and ax are matplotlib objects and avg_probs is numpy array
    """
    # Load data
    results, metadata = load_model_results(model_name, results_dir)
    
    # Get a_an_probs: shape (n_examples, n_layers, 2) where last dim is [a_prob, an_prob]
    a_an_probs = results['a_an_probs']
    
    # Build a filter mask
    mask = pd.Series([True] * len(metadata))
    if article_filter:
        mask &= (metadata['correct_articles'] == article_filter)
    if correct_filter is not None:
        mask &= (metadata['correct?'] == correct_filter)

    # Apply mask
    filtered_metadata = metadata[mask]
    if not len(filtered_metadata):
        raise ValueError("No examples found for the specified filter(s).")
    
    a_an_probs = a_an_probs[mask.values]
    correct_articles = filtered_metadata['correct_articles'].values
    
    # Calculate correct article probabilities
    n_examples, n_layers, _ = a_an_probs.shape
    correct_probs = torch.zeros(n_examples, n_layers)
    
    for i in range(n_examples):
        for layer in range(n_layers):
            if correct_articles[i] == 'a':
                correct_probs[i, layer] = a_an_probs[i, layer, 0]  # a_prob
            else:  # correct_articles[i] == 'an'
                correct_probs[i, layer] = a_an_probs[i, layer, 1]  # an_prob
    
    # Calculate average probability across examples
    avg_probs = correct_probs.mean(dim=0).numpy()
    
    # Create or use provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Reshape for vertical heatmap (n_layers rows, 1 column)
    # Reverse order so lower layers are at bottom, higher at top
    heatmap_data = avg_probs[::-1].reshape(-1, 1)
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                xticklabels=['Avg Correct Article Prob'],
                yticklabels=[f'Layer {i}' for i in range(n_layers-1, -1, -1)],
                cbar_kws={'label': 'Probability'},
                ax=ax)
    
    # Set title
    filter_parts = []
    if article_filter:
        filter_parts.append(f"'{article_filter}' examples")
    if correct_filter is not None:
        filter_parts.append("correct" if correct_filter else "incorrect")
    filter_str = f" ({', '.join(filter_parts)} only)" if filter_parts else ""
    ax.set_title(f'Average Correct Article Probability by Layer\n{model_name}{filter_str}')
    ax.set_xlabel('')
    ax.set_ylabel('Layer')
    
    return fig, ax, avg_probs

def create_example_topk_heatmap(model_name: str, example_idx: int,
                               figsize: Tuple[int, int] = (15, 8),
                               results_dir: str = 'results/logit-lens',
                               ax: Optional[plt.Axes] = None):
    """
    Create a heatmap showing top-k tokens and their probabilities by layer for a specific example
    
    Args:
        model_name: Name of the model to analyze
        example_idx: Index of the example to analyze
        figsize: Figure size tuple (only used if ax is None)
        results_dir: Directory containing the results
        ax: Optional matplotlib axes to plot on
        
    Returns:
        tuple: (fig, ax, heatmap_data, y_labels) where fig and ax are matplotlib objects
    """
    # Load data
    results, metadata = load_model_results(model_name, results_dir)
    
    # Get data for specific example
    top_k_probs = results['top_k_probs'][example_idx]  # shape: (n_layers, k)
    token_strings = results['token_strings'][example_idx]  # list of lists
    
    n_layers, k = top_k_probs.shape
    
    # Create labels for y-axis (tokens by layer)
    y_labels = []
    heatmap_data = []
    
    for layer in range(n_layers):
        layer_probs = top_k_probs[layer].numpy()
        layer_tokens = token_strings[layer]
        
        # Create row for this layer
        heatmap_data.append(layer_probs)
        
        # Create labels showing tokens
        token_labels = [f"'{token}'" for token in layer_tokens]
        y_labels.append(f"L{layer}: {' | '.join(token_labels)}")
    
    # Convert to numpy array
    heatmap_data = np.array(heatmap_data)
    
    # Reverse order so lower layers are at bottom, higher at top
    heatmap_data = heatmap_data[::-1]
    y_labels = y_labels[::-1]
    
    # Create or use provided axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f', 
                cmap='viridis',
                xticklabels=[f'Top-{i+1}' for i in range(k)],
                yticklabels=y_labels,
                cbar_kws={'label': 'Probability'},
                ax=ax)
    
    # Get example info
    profession = metadata.iloc[example_idx]['professions']
    correct_article = metadata.iloc[example_idx]['correct_articles']
    
    ax.set_title(f'Top-{k} Token Probabilities by Layer\n{model_name} - Example {example_idx}: {profession} (correct: {correct_article})')
    ax.set_xlabel('Top-k Rank')
    ax.set_ylabel('Layer: Top Tokens')
    
    return fig, ax, heatmap_data, y_labels

def create_multi_model_comparison(model_names: list, 
                                 article_filter: Optional[str] = None,
                                 correct_filter: Optional[bool] = None,
                                 figsize: Tuple[int, int] = (20, 8),
                                 results_dir: str = 'results/logit-lens'):
    """
    Create a figure comparing article probability heatmaps across multiple models
    
    Args:
        model_names: List of model names to compare
        article_filter: Filter examples by article ('a', 'an', or None for all)
        correct_filter: Filter for correct (True) or incorrect (False) predictions.
        figsize: Figure size tuple
        results_dir: Directory containing the results
        
    Returns:
        tuple: (fig, axes, all_avg_probs) where fig is the figure, axes is array of axes, and all_avg_probs is dict of results
    """
    n_models = len(model_names)
    
    # First pass: get all data and find max layers
    all_avg_probs = {}
    max_layers = 0
    
    for model_name in model_names:
        results, metadata = load_model_results(model_name, results_dir)
        a_an_probs = results['a_an_probs']
        
        # Build a filter mask and apply it
        mask = pd.Series([True] * len(metadata))
        if article_filter:
            mask &= (metadata['correct_articles'] == article_filter)
        if correct_filter is not None:
            mask &= (metadata['correct?'] == correct_filter)
            
        filtered_metadata = metadata[mask]
        if not len(filtered_metadata):
            raise ValueError(f"No examples found for the specified filters in model {model_name}")

        a_an_probs = a_an_probs[mask.values]
        correct_articles = filtered_metadata['correct_articles'].values

        # Calculate correct article probabilities
        n_examples, n_layers, _ = a_an_probs.shape
        correct_probs = torch.zeros(n_examples, n_layers)
        
        for i in range(n_examples):
            for layer in range(n_layers):
                if correct_articles[i] == 'a':
                    correct_probs[i, layer] = a_an_probs[i, layer, 0]  # a_prob
                else:  # correct_articles[i] == 'an'
                    correct_probs[i, layer] = a_an_probs[i, layer, 1]  # an_prob
        
        # Calculate average probability across examples
        avg_probs = correct_probs.mean(dim=0).numpy()
        all_avg_probs[model_name] = avg_probs
        max_layers = max(max_layers, n_layers)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    # Prepare data for shared colorbar
    all_heatmap_data = []
    
    for i, model_name in enumerate(model_names):
        avg_probs = all_avg_probs[model_name]
        n_layers = len(avg_probs)
        
        # Pad shorter models at the top (higher indices)
        if n_layers < max_layers:
            # Add NaN values at the top for missing layers
            padded_probs = np.full(max_layers, np.nan)
            padded_probs[max_layers-n_layers:] = avg_probs[::-1]  # Put real data at the end (bottom of plot)
        else:
            padded_probs = avg_probs[::-1]  # Reverse for bottom-to-top
        
        heatmap_data = padded_probs.reshape(-1, 1)
        all_heatmap_data.append(heatmap_data)
    
    # Find global min/max for shared colorbar (ignoring NaN values)
    all_data = np.concatenate(all_heatmap_data)
    vmin = np.nanmin(all_data)
    vmax = np.nanmax(all_data)
    
    # Create heatmaps
    for i, model_name in enumerate(model_names):
        heatmap_data = all_heatmap_data[i]
        
        # Create y-axis labels only for the first model
        if i == 0:
            # Create labels for all max_layers positions (from top to bottom)
            y_labels = [f'Layer {j}' for j in range(max_layers-1, -1, -1)]
        else:
            y_labels = False  # This will remove y-axis labels entirely
        
        # Create heatmap with shared colorbar range
        sns.heatmap(heatmap_data, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='viridis',
                    xticklabels=[],  # Remove x-axis labels
                    yticklabels=y_labels,
                    cbar=i == n_models - 1,  # Only show colorbar on last subplot
                    vmin=vmin,
                    vmax=vmax,
                    ax=axes[i])
        
        # Set model name at bottom instead of top
        axes[i].set_xlabel(model_name)
        axes[i].set_title('')  # Remove title
        
        # Remove y-axis labels for all but the first subplot
        if i > 0:
            axes[i].set_ylabel('')
    
    # Add shared x-axis label
    fig.text(0.5, 0.02, 'Average Correct Article Probability', ha='center', fontsize=12)
    
    # Add overall title
    filter_parts = []
    if article_filter:
        filter_parts.append(f"'{article_filter}' examples")
    if correct_filter is not None:
        filter_parts.append("correct" if correct_filter else "incorrect")
    filter_str = f" ({', '.join(filter_parts)} only)" if filter_parts else ""
    fig.suptitle(f'Model Comparison: Average Correct Article Probability{filter_str}', fontsize=16, y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)  # Increase top padding and bottom space
    
    return fig, axes, all_avg_probs

#%%
fig, ax, avg_probs = create_article_probability_heatmap('Qwen3-1.7B', article_filter='an')
plt.tight_layout()
plt.show()
#%%
fig, ax, avg_probs = create_article_probability_heatmap('Qwen3-1.7B', article_filter='an')
plt.tight_layout()
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8))
fig.savefig('results/logit-lens/all-models.png')
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8), correct_filter=True)
#fig.savefig('results/logit-lens/all-models.png')
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8), article_filter='a')
fig.savefig('results/logit-lens/all-models-a.png')
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8), article_filter='a', correct_filter=True)
#fig.savefig('results/logit-lens/all-models-a.png')
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8), article_filter='an')
fig.savefig('results/logit-lens/all-models-an.png')
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8), article_filter='an', correct_filter=True)
fig.savefig('results/logit-lens/all-models-an-correct.png')
plt.show()
#%%
fig, axes, all_probs = create_multi_model_comparison(["Qwen3-0.6B","Qwen3-1.7B", "Qwen3-4B","Qwen3-8B","Qwen3-14B"][::-1], 
                                                    figsize=(10,8), article_filter='an', correct_filter=False)
fig.savefig('results/logit-lens/all-models-an-incorrect.png')
plt.show()
# %%
results, metadata = load_model_results("Qwen3-0.6B")
#%%
an_correct = metadata['correct_articles'].values == 'an'
# %%
results['a_an_probs'][an_correct, 19, 1]
# %%
sentences = metadata[an_correct]['sentences'].tolist()
# %%
from transformer_lens import HookedTransformer 
model = HookedTransformer.from_pretrained("Qwen3-0.6B")
# %%
all_reps = [[] for _ in range(model.cfg.n_layers)]
for sentence in sentences:
    _, cache = model.run_with_cache(sentence)
    for layer in range(model.cfg.n_layers):
        reps = cache[f'blocks.{layer}.hook_resid_post'].squeeze(0)[-1]
        all_reps[layer].append(reps)
all_reps = [torch.stack(x) for x in all_reps]
all_reps = torch.stack(all_reps)
# %%
logits = model.ln_final(all_reps) @ model.W_U
probs = torch.softmax(logits, dim=-1)
# %%
an_token = model.tokenizer(' an', add_special_tokens=False)['input_ids'][0]
# %%
probs[:, :, an_token].mean(1)
# %%
