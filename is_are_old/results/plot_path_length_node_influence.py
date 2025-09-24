#%%
# Import additional required modules
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from circuit_tracer.graph import Graph, normalize_matrix

# Define threshold parameters

LAST_ONLY = True
PATH_LENGTH_RESULTS_DIR = Path('path_length_last_new') if LAST_ONLY else Path('path_length_new')  # Directory for path length results

models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']
model_sizes = [28,28, 36, 36, 40]
#%%
# Load results from disk for plotting
def load_saved_results(models):
    """Load saved results from disk"""
    loaded_results = {}
    
    for model in models:
        # Convert model name format from qwen3-0.6b-relu-lowl0 to Qwen3-0.6B
        model_name_parts = model.split('-')
        size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
        if size_part.endswith('B'):
            size_part = size_part[:-1] + 'B'
        logit_lens_model_name = f"Qwen3-{size_part}"
        
        results_file = PATH_LENGTH_RESULTS_DIR / f'{logit_lens_model_name}_path_length_results.pt'
        loaded_results[model] = torch.load(results_file, weights_only=False)
        #loaded_results[model]['extra_metadata'] = pd.read_csv(f'interventions_last/{logit_lens_model_name}.csv')
        print(f"Loaded results for {model} from {results_file}")
    
    return loaded_results

# Load results from disk
all_model_results = load_saved_results(models)
# %%
# Plot all models on same plot - by article type (zoomed to 0-10)
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Define colors for each model
colors = ['blue', 'red', 'green', 'orange', 'purple']
# Define line styles for each subcategory
line_styles = {
    'all': '-',      # solid
    'are': '--',       # dashed  
    'is': ':'        # dotted
}

# Create clean model names
def get_clean_model_name(model):
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    return f"Qwen3-{size_part}"


def augment_metadata(metadata: pd.DataFrame, model_name):
    performance_df = pd.read_csv(f'is-are-animals-repeat/{model_name}.csv')
    # Align performance_df with metadata based on prompt columns
    # metadata has "prompt" (lowercase), performance_df has "Prompt" (capitalized)
    
    # Filter to only include prompts that exist in metadata
    metadata_prompts = set(metadata['prompt'].values)
    
    # Filter performance_df to only include rows that have corresponding entries in metadata
    performance_df_filtered = performance_df[performance_df['Prompt'].isin(metadata_prompts)].copy()
    
    # Sort performance_df to match the order of metadata
    performance_df_aligned = performance_df_filtered.set_index('Prompt').loc[metadata['prompt']].reset_index()
    
    # Add the aligned performance data to metadata
    metadata['correct?'] = performance_df_aligned['Verb_Correct'].values

    return metadata
#%%
# Plot relationship between selected node influence and model performance
def plot_selected_influence_vs_performance(model_results, model_name):
    """Create scatter plots for selected node influence vs performance"""
    metadata = model_results['metadata']
    cumsum_selected = model_results['per_example_cumsum_selected_influences']
    final_selected_influence = cumsum_selected[:, -1]  # Get the last cumsum value for each example
    
    # Get logit probabilities from metadata
    logit_probs = torch.tensor(metadata['logit_probability'].values)
    
    # Split by article type
    are_mask = metadata['answer'] == 'are'
    is_mask = metadata['answer'] == 'is'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for 'are' examples
    are_influences = final_selected_influence[are_mask].cpu().numpy()
    are_probs = logit_probs[are_mask].cpu().numpy()
    ax1.scatter(are_influences, are_probs, alpha=0.5)
    ax1.set_xlabel('Final Cumulative Selected Node Influence')
    ax1.set_ylabel('Probability of Correct Article')
    ax1.set_title(f'{model_name} - "are" Examples')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    are_corr = np.corrcoef(are_influences, are_probs)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {are_corr:.3f}', 
             transform=ax1.transAxes, verticalalignment='top')
    
    # Plot for 'is' examples
    is_influences = final_selected_influence[is_mask].cpu().numpy()
    is_probs = logit_probs[is_mask].cpu().numpy()
    ax2.scatter(is_influences, is_probs, alpha=0.5)
    ax2.set_xlabel('Final Cumulative Selected Node Influence')
    ax2.set_ylabel('Probability of Correct is/are')
    ax2.set_title(f'{model_name} - "is" Examples')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    is_corr = np.corrcoef(is_influences, is_probs)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {is_corr:.3f}', 
             transform=ax2.transAxes, verticalalignment='top')
    
    plt.suptitle(f'Selected Node Influence vs Performance - {model_name}')
    plt.tight_layout()
    plt.show()
    
    return are_corr, is_corr

# # Plot for each model and collect correlations
# correlations = {}
# for model in models:
#     clean_name = get_clean_model_name(model)
#     a_corr, an_corr = plot_selected_influence_vs_performance(all_model_results[model], clean_name)
#     correlations[clean_name] = {'a': a_corr, 'an': an_corr}

# #%%
# # Plot correlation coefficients across models
# fig, ax = plt.subplots(figsize=(10, 6))

# x = np.arange(len(correlations))
# width = 0.35

# a_corrs = [corr['a'] for corr in correlations.values()]
# an_corrs = [corr['an'] for corr in correlations.values()]

# ax.bar(x - width/2, a_corrs, width, label='"a" examples')
# ax.bar(x + width/2, an_corrs, width, label='"an" examples')

# ax.set_ylabel('Correlation Coefficient')
# ax.set_title('Correlation between Selected Node Influence and Performance')
# ax.set_xticks(x)
# ax.set_xticklabels(list(correlations.keys()), rotation=45)
# ax.legend()
# ax.grid(True, alpha=0.3)

# plt.tight_layout()

# %%
# Create scatter plots showing cumulative influence at max path length by model
def create_max_influence_scatter_plots(all_model_results, models):
    """Create scatter plots showing cumulative influence at max path length for each model"""
    
    # Extract data for each model
    model_names = []
    a_correct_data = []
    a_incorrect_data = []
    an_correct_data = []
    an_incorrect_data = []
    
    for model_idx, model in enumerate(models):
        clean_name = get_clean_model_name(model)
        model_names.append(clean_name)
        results = all_model_results[model]
        metadata = results['metadata']
        metadata = augment_metadata(metadata, clean_name)
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each category using the correct method
        a_corrects = per_example_influences[(metadata['correct?']) & (metadata['answer'] == 'are')]
        a_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['answer'] == 'are')]
        an_corrects = per_example_influences[(metadata['correct?']) & (metadata['answer'] == 'is')]
        an_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['answer'] == 'is')]
        
        # Store the individual example influences with model indices for scatter plotting
        for influence in a_corrects.cpu().numpy():
            a_correct_data.append((model_idx, influence))
        for influence in a_incorrects.cpu().numpy():
            a_incorrect_data.append((model_idx, influence))
        for influence in an_corrects.cpu().numpy():
            an_correct_data.append((model_idx, influence))
        for influence in an_incorrects.cpu().numpy():
            an_incorrect_data.append((model_idx, influence))
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for 'are' examples
    if a_correct_data:
        a_correct_x, a_correct_y = zip(*a_correct_data)
        ax1.scatter(a_correct_x, a_correct_y, color='green', s=50, alpha=0.6, label='Correct')
    if a_incorrect_data:
        a_incorrect_x, a_incorrect_y = zip(*a_incorrect_data)
        ax1.scatter(a_incorrect_x, a_incorrect_y, color='red', s=50, alpha=0.6, label='Incorrect')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Cumulative Influence at Max Path Length')
    ax1.set_title('"are" Examples - Selected Node Influence')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for 'is' examples
    if an_correct_data:
        an_correct_x, an_correct_y = zip(*an_correct_data)
        ax2.scatter(an_correct_x, an_correct_y, color='green', s=50, alpha=0.6, label='Correct')
    if an_incorrect_data:
        an_incorrect_x, an_incorrect_y = zip(*an_incorrect_data)
        ax2.scatter(an_incorrect_x, an_incorrect_y, color='red', s=50, alpha=0.6, label='Incorrect')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Cumulative Influence at Max Path Length')
    ax2.set_title('"is" Examples - Selected Node Influence')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Cumulative Influence at Max Path Length by Model', fontsize=16)
    plt.tight_layout()
    plt.show()

# Create the scatter plots
create_max_influence_scatter_plots(all_model_results, models)

# %%
# Create scatter plots showing MEAN cumulative influence at max path length by model
def create_mean_max_influence_scatter_plots(all_model_results, models):
    """Create scatter plots showing mean cumulative influence at max path length for each model"""
    
    # Extract data for each model
    model_names = []
    a_correct_means = []
    a_incorrect_means = []
    an_correct_means = []
    an_incorrect_means = []
    
    for model_idx, model in enumerate(models):
        clean_name = get_clean_model_name(model)
        model_names.append(clean_name)
        results = all_model_results[model]
        metadata = results['metadata']
        metadata = augment_metadata(metadata, clean_name)
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each category using the correct method
        a_corrects = per_example_influences[(metadata['correct?']) & (metadata['answer'] == 'are')]
        a_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['answer'] == 'are')]
        an_corrects = per_example_influences[(metadata['correct?']) & (metadata['answer'] == 'is')]
        an_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['answer'] == 'is')]
        
        # Calculate means for each category
        a_correct_means.append(a_corrects.mean().item() if len(a_corrects) > 0 else 0)
        a_incorrect_means.append(a_incorrects.mean().item() if len(a_incorrects) > 0 else 0)
        an_correct_means.append(an_corrects.mean().item() if len(an_corrects) > 0 else 0)
        an_incorrect_means.append(an_incorrects.mean().item() if len(an_incorrects) > 0 else 0)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for 'are' examples
    x_pos = np.arange(len(model_names))
    ax1.scatter(x_pos, a_correct_means, color='green', s=100, alpha=0.8, label='Correct (mean)')
    ax1.scatter(x_pos, a_incorrect_means, color='red', s=100, alpha=0.8, label='Incorrect (mean)')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Mean Cumulative Influence at Max Path Length')
    ax1.set_title('"are" Examples - Mean Selected Node Influence')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for 'is' examples
    ax2.scatter(x_pos, an_correct_means, color='green', s=100, alpha=0.8, label='Correct (mean)')
    ax2.scatter(x_pos, an_incorrect_means, color='red', s=100, alpha=0.8, label='Incorrect (mean)')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Mean Cumulative Influence at Max Path Length')
    ax2.set_title('"is" Examples - Mean Selected Node Influence')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Mean Cumulative Influence at Max Path Length by Model', fontsize=16)
    plt.tight_layout()
    plt.show()

# Create the mean scatter plots
create_mean_max_influence_scatter_plots(all_model_results, models)

# %%
# Create bar plot with 4 categories: correct/incorrect x is/are
def create_four_category_bar_plot(all_model_results, models):
    """Create bar plot showing mean cumulative influence for 4 categories per model"""
    
    # Extract data for each model
    model_names = []
    are_correct_means = []
    are_incorrect_means = []
    is_correct_means = []
    is_incorrect_means = []
    
    for model_idx, model in enumerate(models):
        clean_name = get_clean_model_name(model)
        model_names.append(clean_name)
        results = all_model_results[model]
        metadata = results['metadata']
        metadata = augment_metadata(metadata, clean_name)
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each of the 4 categories
        are_corrects = per_example_influences[(metadata['correct?']) & (metadata['answer'] == 'are')]
        are_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['answer'] == 'are')]
        is_corrects = per_example_influences[(metadata['correct?']) & (metadata['answer'] == 'is')]
        is_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['answer'] == 'is')]
        
        # Calculate means for each category
        are_correct_means.append(are_corrects.mean().item() if len(are_corrects) > 0 else 0)
        are_incorrect_means.append(are_incorrects.mean().item() if len(are_incorrects) > 0 else 0)
        is_correct_means.append(is_corrects.mean().item() if len(is_corrects) > 0 else 0)
        is_incorrect_means.append(is_incorrects.mean().item() if len(is_incorrects) > 0 else 0)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.2
    
    # Create bars for each category
    # Steel blue for correct, dark red for incorrect
    # Solid bars for 'are', dotted bars for 'is'
    bars1 = ax.bar(x - 1.5*width, are_correct_means, width, 
                   color='steelblue', alpha=0.8, label='"are" Correct', 
                   edgecolor='black', linewidth=0.5)
    
    bars2 = ax.bar(x - 0.5*width, are_incorrect_means, width,
                   color='darkred', alpha=0.8, label='"are" Incorrect',
                   edgecolor='black', linewidth=0.5)
    
    bars3 = ax.bar(x + 0.5*width, is_correct_means, width,
                   color='steelblue', alpha=0.8, label='"is" Correct',
                   hatch='.', edgecolor='black', linewidth=0.5)
    
    bars4 = ax.bar(x + 1.5*width, is_incorrect_means, width,
                   color='darkred', alpha=0.8, label='"is" Incorrect', 
                   hatch='.', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=20)
    ax.set_ylabel('Prop. of Influence Through Selected Nodes', fontsize=20)
    ax.set_title('Mean Proportion of Influence Through Selected Nodes by Model', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("influence_bar_chart_is_are.pdf")
    plt.show()

# Create the four-category bar plot
create_four_category_bar_plot(all_model_results, models)

# %%
