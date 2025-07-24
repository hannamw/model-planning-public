#%%
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List
from tqdm import tqdm

models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']

def load_model_influences(model_name: str, metadata_dir: str = 'results/logit-lens', path_len_results_dir='results/path_length') -> Dict:
    """Load path length influences for a specific model"""
    metadata_dir = Path(metadata_dir)
    path_len_results_dir = Path(path_len_results_dir)
    
    # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
    model_name_parts = model_name.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    logit_lens_model_name = f"Qwen3-{size_part}"
    
    # Load the main results from .pt file
    results_file = path_len_results_dir / f'{logit_lens_model_name}_path_length_results.pt'
    results = torch.load(results_file, weights_only=False)
    
    # Load the complete metadata from CSV file (which has p(a) and p(an) columns)
    metadata_file = metadata_dir / logit_lens_model_name / 'metadata.csv'
    metadata = pd.read_csv(metadata_file)
    
    # Replace the metadata in results with the complete CSV metadata
    results['metadata'] = metadata
    
    return results

def compute_mean_path_length(path_influences: torch.Tensor) -> torch.Tensor:
    """
    Compute mean path length for each example, weighted by path influence.
    
    Args:
        path_influences: Shape (n_examples, n_path_lengths) - influence at each path length
        
    Returns:
        mean_path_lengths: Shape (n_examples,) - weighted mean path length for each example
    """
    n_examples, n_path_lengths = path_influences.shape
    
    # Path lengths go from 1 to n_path_lengths
    path_lengths = torch.arange(1, n_path_lengths + 1, dtype=torch.float32)
    
    # Compute weighted mean for each example
    # Handle case where total influence is 0
    total_influence = path_influences.sum(dim=1, keepdim=True)
    total_influence = torch.where(total_influence == 0, torch.ones_like(total_influence), total_influence)
    
    weighted_sum = (path_influences * path_lengths.unsqueeze(0)).sum(dim=1)
    mean_path_lengths = weighted_sum / total_influence.squeeze()
    
    return mean_path_lengths

def filter_examples_by_path_length(mean_path_lengths: torch.Tensor, 
                                 target_path_length: float,
                                 window: float = 0.5) -> torch.Tensor:
    """
    Filter examples by mean path length using rounding approach.
    
    Args:
        mean_path_lengths: Mean path length for each example
        target_path_length: Target path length (e.g., 4.0)
        window: Half-window around target (default 0.5 for rounding)
        
    Returns:
        indices: Boolean mask for examples in the target range
    """
    lower_bound = target_path_length - window
    upper_bound = target_path_length + window
    mask = (mean_path_lengths >= lower_bound) & (mean_path_lengths < upper_bound)
    return mask

def get_path_length_range(mean_path_lengths: torch.Tensor) -> Tuple[int, int]:
    """Get the range of rounded path lengths in the data"""
    min_pl = int(torch.floor(mean_path_lengths.min()).item())
    max_pl = int(torch.ceil(mean_path_lengths.max()).item())
    return min_pl, max_pl

def get_examples_at_path_length(metadata: pd.DataFrame, 
                               mean_path_lengths: torch.Tensor,
                               target_path_length: float,
                               article_filter: str | None = None) -> pd.DataFrame:
    """
    Get examples that round to a specific path length.
    
    Args:
        metadata: Metadata dataframe
        mean_path_lengths: Mean path length for each example
        target_path_length: Target path length (e.g., 4.0)
        article_filter: Either 'a', 'an', or None for all examples
        
    Returns:
        filtered_metadata: Examples that round to the target path length
    """
    # First filter by article type if specified
    if article_filter is not None:
        article_mask = metadata['correct_articles'] == article_filter
        filtered_metadata = metadata[article_mask].reset_index(drop=True)
        filtered_path_lengths = mean_path_lengths[article_mask]
    else:
        filtered_metadata = metadata
        filtered_path_lengths = mean_path_lengths
    
    # Then filter by path length
    path_mask = filter_examples_by_path_length(filtered_path_lengths, target_path_length)
    final_metadata = filtered_metadata[path_mask].copy()
    final_metadata['mean_path_length'] = filtered_path_lengths[path_mask].numpy()
    
    return final_metadata

def compute_accuracy_by_path_length(metadata: pd.DataFrame,
                                  mean_path_lengths: torch.Tensor,
                                  article_filter: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute accuracy statistics conditioned on mean path length.
    
    Args:
        metadata: Metadata with 'correct?' and 'correct_articles' columns
        mean_path_lengths: Mean path length for each example
        article_filter: Either 'a', 'an', or None for all examples
        
    Returns:
        path_lengths: Array of path length values (integers)
        accuracies: Accuracy at each path length
        bin_counts: Number of examples at each path length
    """
    # Filter by article type if specified
    if article_filter is not None:
        mask = metadata['correct_articles'] == article_filter
        filtered_metadata = metadata[mask].reset_index(drop=True)
        filtered_path_lengths = mean_path_lengths[mask]
    else:
        filtered_metadata = metadata
        filtered_path_lengths = mean_path_lengths
    
    if len(filtered_metadata) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Get range of path lengths to analyze
    min_pl, max_pl = get_path_length_range(filtered_path_lengths)
    
    path_lengths = []
    accuracies = []
    bin_counts = []
    
    for target_pl in range(min_pl, max_pl + 1):
        mask = filter_examples_by_path_length(filtered_path_lengths, float(target_pl))
        indices = torch.where(mask)[0]
        
        if len(indices) == 0:
            continue
            
        bin_metadata = filtered_metadata.iloc[indices]
        accuracy = bin_metadata['correct?'].mean()
        count = len(indices)
        
        path_lengths.append(target_pl)
        accuracies.append(accuracy)
        bin_counts.append(count)
    
    return np.array(path_lengths), np.array(accuracies), np.array(bin_counts)

def get_correct_article_probabilities(metadata: pd.DataFrame) -> np.ndarray:
    """
    Extract the probability of the correct article for each example.
    
    Args:
        metadata: Metadata DataFrame with 'correct_articles', 'p(a)', and 'p(an)' columns
        
    Returns:
        probabilities: Array of correct article probabilities for each example
    """
    probabilities = []
    for _, row in metadata.iterrows():
        if row['correct_articles'] == 'a':
            probabilities.append(row['p(a)'])
        else:  # correct_articles == 'an'
            probabilities.append(row['p(an)'])
    
    return np.array(probabilities)

def plot_accuracy_vs_path_length(all_results: Dict, models: List[str]):
    """Scatter plot of individual examples: correct article probability vs mean path length"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    def get_clean_model_name(model):
        model_name_parts = model.split('-')
        size_part = model_name_parts[1].upper()
        if size_part.endswith('B'):
            size_part = size_part[:-1] + 'B'
        return f"Qwen3-{size_part}"
    
    # Plot 1: All examples for each model
    ax = axes[0]
    for i, model in enumerate(models):
        results = all_results[model]
        metadata = results['metadata']
        path_influences = results['per_example_path_influences']
        mean_path_lengths = compute_mean_path_length(path_influences)
        
        correct_probs = get_correct_article_probabilities(metadata)
        
        ax.scatter(mean_path_lengths, correct_probs, 
                  color=colors[i], alpha=0.6, s=20, 
                  label=get_clean_model_name(model))
    
    ax.set_xlabel('Mean Path Length')
    ax.set_ylabel('Correct Article Probability')
    ax.set_title('Individual Examples: Correct Article Probability vs Mean Path Length (All)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: 'a' examples for each model
    ax = axes[1]
    for i, model in enumerate(models):
        results = all_results[model]
        metadata = results['metadata']
        path_influences = results['per_example_path_influences']
        mean_path_lengths = compute_mean_path_length(path_influences)
        
        # Filter for 'a' examples
        a_mask = metadata['correct_articles'] == 'a'
        if a_mask.sum() > 0:
            filtered_metadata = metadata[a_mask]
            correct_probs = get_correct_article_probabilities(filtered_metadata)
            filtered_path_lengths = mean_path_lengths[a_mask]
            
            ax.scatter(filtered_path_lengths, correct_probs, 
                      color=colors[i], alpha=0.6, s=20,
                      label=get_clean_model_name(model))
    
    ax.set_xlabel('Mean Path Length')
    ax.set_ylabel('Correct Article Probability')
    ax.set_title('Individual Examples: Correct Article Probability vs Mean Path Length ("a" Examples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: 'an' examples for each model
    ax = axes[2]
    for i, model in enumerate(models):
        results = all_results[model]
        metadata = results['metadata']
        path_influences = results['per_example_path_influences']
        mean_path_lengths = compute_mean_path_length(path_influences)
        
        # Filter for 'an' examples
        an_mask = metadata['correct_articles'] == 'an'
        if an_mask.sum() > 0:
            filtered_metadata = metadata[an_mask]
            correct_probs = get_correct_article_probabilities(filtered_metadata)
            filtered_path_lengths = mean_path_lengths[an_mask]
            
            ax.scatter(filtered_path_lengths, correct_probs, 
                      color=colors[i], alpha=0.6, s=20,
                      label=get_clean_model_name(model))
    
    ax.set_xlabel('Mean Path Length')
    ax.set_ylabel('Correct Article Probability')
    ax.set_title('Individual Examples: Correct Article Probability vs Mean Path Length ("an" Examples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 4-6: Individual model breakdowns
    for model_idx, model in enumerate(models[:3]):  # Show first 3 models
        ax = axes[3 + model_idx]
        results = all_results[model]
        metadata = results['metadata']
        path_influences = results['per_example_path_influences']
        mean_path_lengths = compute_mean_path_length(path_influences)
        
        # Plot all, 'a', and 'an' for this model
        for article_type, color, label in [('all', 'black', 'All'), ('a', 'red', "'a'"), ('an', 'green', "'an'")]:
            if article_type == 'all':
                mask = pd.Series([True] * len(metadata))
                filtered_metadata = metadata
                filtered_path_lengths = mean_path_lengths
            else:
                mask = metadata['correct_articles'] == article_type
                filtered_metadata = metadata[mask]
                filtered_path_lengths = mean_path_lengths[mask]
            
            if mask.sum() > 0:
                correct_probs = get_correct_article_probabilities(filtered_metadata)
                
                ax.scatter(filtered_path_lengths, correct_probs, 
                          color=color, alpha=0.6, s=20, label=label)
        
        ax.set_xlabel('Mean Path Length')
        ax.set_ylabel('Correct Article Probability')
        ax.set_title(f'{get_clean_model_name(model)} Breakdown')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()


all_results = {}

for model in tqdm(models, desc="Loading models"):
    all_results[model] = load_model_influences(model)

print("\nComputing mean path length statistics...")
for model in models:
    results = all_results[model]
    metadata = results['metadata']
    path_influences = results['per_example_path_influences']
    mean_path_lengths = compute_mean_path_length(path_influences)
    
    print(f"\n{model}:")
    print(f"  Mean path length (all): {mean_path_lengths.mean():.2f} ± {mean_path_lengths.std():.2f}")
    
    # By article type
    is_a = metadata['correct_articles'] == 'a'
    is_an = metadata['correct_articles'] == 'an'
    
    if is_a.sum() > 0:
        print(f"  Mean path length ('a'): {mean_path_lengths[is_a].mean():.2f} ± {mean_path_lengths[is_a].std():.2f}")
    if is_an.sum() > 0:
        print(f"  Mean path length ('an'): {mean_path_lengths[is_an].mean():.2f} ± {mean_path_lengths[is_an].std():.2f}")
    
    # By correctness
    is_correct = metadata['correct?'] == True
    is_incorrect = ~is_correct
    
    if is_correct.sum() > 0:
        print(f"  Mean path length (correct): {mean_path_lengths[is_correct].mean():.2f} ± {mean_path_lengths[is_correct].std():.2f}")
    if is_incorrect.sum() > 0:
        print(f"  Mean path length (incorrect): {mean_path_lengths[is_incorrect].mean():.2f} ± {mean_path_lengths[is_incorrect].std():.2f}")

print("\nPlotting results...")
plot_accuracy_vs_path_length(all_results, models)
    

# %%
def plot_mean_path_length_vs_correctness_an_only(all_results: Dict, models: List[str]):
    """Scatter plot: mean path length vs correct article probability for all models, conditioned on answer='an'"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    def get_clean_model_name(model):
        model_name_parts = model.split('-')
        size_part = model_name_parts[1].upper()
        if size_part.endswith('B'):
            size_part = size_part[:-1] + 'B'
        return f"Qwen3-{size_part}"
    
    # Plot each model
    for i, model in enumerate(models):
        results = all_results[model]
        metadata = results['metadata']
        path_influences = results['per_example_path_influences']
        mean_path_lengths = compute_mean_path_length(path_influences)
        
        # Filter for 'an' examples only
        an_mask = metadata['correct_articles'] == 'an'
        if an_mask.sum() > 0:
            filtered_metadata = metadata[an_mask]
            correct_probs = get_correct_article_probabilities(filtered_metadata)
            filtered_path_lengths = mean_path_lengths[an_mask]
            
            ax.scatter(filtered_path_lengths, correct_probs, 
                      color=colors[i], alpha=0.7, s=30,
                      label=f"{get_clean_model_name(model)} (n={len(filtered_path_lengths)})")
    
    ax.set_xlabel('Mean Path Length', fontsize=12)
    ax.set_ylabel('Correct Article Probability', fontsize=12)
    ax.set_title('Mean Path Length vs. Correct Article Probability\n(Conditioned on Answer = "an")', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

print("\nCreating scatter plot for 'an' examples...")
plot_mean_path_length_vs_correctness_an_only(all_results, models)

# %%
results = torch.load('results/path_length/Qwen3-0.6B_path_length_results.pt', weights_only=False)
# %%
results.keys()

# %%
