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
        loaded_results[model]['extra_metadata'] = pd.read_csv(f'interventions_last/{logit_lens_model_name}.csv')
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
    'a': '--',       # dashed  
    'an': ':'        # dotted
}

# Create clean model names
def get_clean_model_name(model):
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    return f"Qwen3-{size_part}"
#%%
# Plot selected vs non-selected path influences for each model
for model in models:
    clean_name = get_clean_model_name(model)
    results = all_model_results[model]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    x_range = range(1, len(results['all']['total']) + 1)
    
    # Plot for 'a' correct examples
    ax1.plot(x_range, results['a_correct']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax1.plot(x_range, results['a_correct']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax1.set_title(f'"a" Correct Examples')
    ax1.set_xlabel('Path Length')
    ax1.set_ylabel('Cumulative Path Influence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for 'a' incorrect examples
    ax2.plot(x_range, results['a_incorrect']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax2.plot(x_range, results['a_incorrect']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax2.set_title(f'"a" Incorrect Examples')
    ax2.set_xlabel('Path Length')
    ax2.set_ylabel('Cumulative Path Influence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot for 'an' correct examples
    ax3.plot(x_range, results['an_correct']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax3.plot(x_range, results['an_correct']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax3.set_title(f'"an" Correct Examples')
    ax3.set_xlabel('Path Length')
    ax3.set_ylabel('Cumulative Path Influence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot for 'an' incorrect examples
    ax4.plot(x_range, results['an_incorrect']['selected'].cpu().numpy(), 'g-', label='Selected', linewidth=2)
    ax4.plot(x_range, results['an_incorrect']['non_selected'].cpu().numpy(), 'r-', label='Non-selected', linewidth=2)
    ax4.set_title(f'"an" Incorrect Examples')
    ax4.set_xlabel('Path Length')
    ax4.set_ylabel('Cumulative Path Influence')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Selected vs Non-selected Path Influences - {clean_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

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
    is_a = metadata['correct_articles'] == 'a'
    is_an = metadata['correct_articles'] == 'an'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for 'a' examples
    a_influences = final_selected_influence[is_a].cpu().numpy()
    a_probs = logit_probs[is_a].cpu().numpy()
    ax1.scatter(a_influences, a_probs, alpha=0.5)
    ax1.set_xlabel('Final Cumulative Selected Node Influence')
    ax1.set_ylabel('Probability of Correct Article')
    ax1.set_title(f'{model_name} - "a" Examples')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    a_corr = np.corrcoef(a_influences, a_probs)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {a_corr:.3f}', 
             transform=ax1.transAxes, verticalalignment='top')
    
    # Plot for 'an' examples
    an_influences = final_selected_influence[is_an].cpu().numpy()
    an_probs = logit_probs[is_an].cpu().numpy()
    ax2.scatter(an_influences, an_probs, alpha=0.5)
    ax2.set_xlabel('Final Cumulative Selected Node Influence')
    ax2.set_ylabel('Probability of Correct Article')
    ax2.set_title(f'{model_name} - "an" Examples')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    an_corr = np.corrcoef(an_influences, an_probs)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {an_corr:.3f}', 
             transform=ax2.transAxes, verticalalignment='top')
    
    plt.suptitle(f'Selected Node Influence vs Performance - {model_name}')
    plt.tight_layout()
    plt.show()
    
    return a_corr, an_corr

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

#%%
# Create aggregated plots across models
def create_aggregated_plot(article_type, all_model_results, models, xlim=None, model_sizes=None, to_plot=['selected', 'non_selected']):
    """Create aggregated plot for a specific article type ('a' or 'an')"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define line styles for different categories
    line_styles = {
        ('selected', True): '-',      # solid
        ('selected', False): '--',    # dashed
        ('non_selected', True): ':',  # dotted
        ('non_selected', False): '-.' # dash-dot
    }
    
    # Define colors for each model
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']  # Classic distinct colors
    
    lines = []  # Store lines for legend
    labels = []  # Store labels for legend
    
    for model_idx, model in enumerate(models):
        clean_name = get_clean_model_name(model)
        results = all_model_results[model]
        color = colors[model_idx]
        
        x_range = np.arange(1, len(results['all']['total']) + 1)
        if model_sizes:
            x_range = x_range / model_sizes[model_idx]
        
        # Plot selected and non-selected for both correct and incorrect cases
        for influence_type in to_plot:
            for is_correct in [True, False]:
                key = f"{article_type}_{'correct' if is_correct else 'incorrect'}"
                data = results[key][influence_type].cpu().numpy()
                
                line = ax.plot(x_range, data, 
                             line_styles[(influence_type, is_correct)],
                             color=color,
                             label=f"{clean_name} - {influence_type} ({'correct' if is_correct else 'incorrect'})",
                             linewidth=2)
                
                lines.append(line[0])
                labels.append(f"{clean_name} - {influence_type} ({'correct' if is_correct else 'incorrect'})")
    
    ax.set_xlabel('Path Length')
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylabel('Cumulative Path Influence')
    ax.set_title(f'Aggregated Path Influences - "{article_type}" Examples')
    ax.grid(True, alpha=0.3)
    
    # Create legend at the bottom
    ax.legend(lines, labels, 
             loc='upper center', 
             bbox_to_anchor=(0.5, -0.15),
             ncol=5,
             fontsize='small')
    
    plt.tight_layout()
    plt.show()

# Create separate plots for 'a' and 'an'
create_aggregated_plot('a', all_model_results, models)
create_aggregated_plot('an', all_model_results, models)
create_aggregated_plot('a', all_model_results, models, xlim=(0,12))
create_aggregated_plot('an', all_model_results, models, xlim=(0,12))

# %%
create_aggregated_plot('a', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes)
create_aggregated_plot('an', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes)
# %%
create_aggregated_plot('a', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes, to_plot=['selected'])
create_aggregated_plot('an', all_model_results, models, xlim=(0,0.4), model_sizes=model_sizes, to_plot=['selected'])

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
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each category using the correct method
        a_corrects = per_example_influences[(metadata['correct?']) & (metadata['correct_articles'] == 'a')]
        a_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['correct_articles'] == 'a')]
        an_corrects = per_example_influences[(metadata['correct?']) & (metadata['correct_articles'] == 'an')]
        an_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['correct_articles'] == 'an')]
        
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
    
    # Plot for 'a' examples
    if a_correct_data:
        a_correct_x, a_correct_y = zip(*a_correct_data)
        ax1.scatter(a_correct_x, a_correct_y, color='green', s=50, alpha=0.6, label='Correct')
    if a_incorrect_data:
        a_incorrect_x, a_incorrect_y = zip(*a_incorrect_data)
        ax1.scatter(a_incorrect_x, a_incorrect_y, color='red', s=50, alpha=0.6, label='Incorrect')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Cumulative Influence at Max Path Length')
    ax1.set_title('"a" Examples - Selected Node Influence')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for 'an' examples
    if an_correct_data:
        an_correct_x, an_correct_y = zip(*an_correct_data)
        ax2.scatter(an_correct_x, an_correct_y, color='green', s=50, alpha=0.6, label='Correct')
    if an_incorrect_data:
        an_incorrect_x, an_incorrect_y = zip(*an_incorrect_data)
        ax2.scatter(an_incorrect_x, an_incorrect_y, color='red', s=50, alpha=0.6, label='Incorrect')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Cumulative Influence at Max Path Length')
    ax2.set_title('"an" Examples - Selected Node Influence')
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
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each category using the correct method
        a_corrects = per_example_influences[(metadata['correct?']) & (metadata['correct_articles'] == 'a')]
        a_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['correct_articles'] == 'a')]
        an_corrects = per_example_influences[(metadata['correct?']) & (metadata['correct_articles'] == 'an')]
        an_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['correct_articles'] == 'an')]
        
        # Calculate means for each category
        a_correct_means.append(a_corrects.mean().item() if len(a_corrects) > 0 else 0)
        a_incorrect_means.append(a_incorrects.mean().item() if len(a_incorrects) > 0 else 0)
        an_correct_means.append(an_corrects.mean().item() if len(an_corrects) > 0 else 0)
        an_incorrect_means.append(an_incorrects.mean().item() if len(an_incorrects) > 0 else 0)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for 'a' examples
    x_pos = np.arange(len(model_names))
    ax1.scatter(x_pos, a_correct_means, color='green', s=100, alpha=0.8, label='Correct (mean)')
    ax1.scatter(x_pos, a_incorrect_means, color='red', s=100, alpha=0.8, label='Incorrect (mean)')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Mean Cumulative Influence at Max Path Length')
    ax1.set_title('"a" Examples - Mean Selected Node Influence')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot for 'an' examples
    ax2.scatter(x_pos, an_correct_means, color='green', s=100, alpha=0.8, label='Correct (mean)')
    ax2.scatter(x_pos, an_incorrect_means, color='red', s=100, alpha=0.8, label='Incorrect (mean)')
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Mean Cumulative Influence at Max Path Length')
    ax2.set_title('"an" Examples - Mean Selected Node Influence')
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
# Create combined bar plot showing mean cumulative influence at max path length
def create_combined_mean_influence_bar_plot(all_model_results, models):
    """Create combined bar plot showing mean cumulative influence with colors for correct/incorrect and patterns for a/an"""
    
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
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each category using the correct method
        a_corrects = per_example_influences[(metadata['correct?']) & (metadata['correct_articles'] == 'a')]
        a_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['correct_articles'] == 'a')]
        an_corrects = per_example_influences[(metadata['correct?']) & (metadata['correct_articles'] == 'an')]
        an_incorrects = per_example_influences[(~metadata['correct?']) & (metadata['correct_articles'] == 'an')]
        
        # Calculate means for each category
        a_correct_means.append(a_corrects.mean().item() if len(a_corrects) > 0 else 0)
        a_incorrect_means.append(a_incorrects.mean().item() if len(a_incorrects) > 0 else 0)
        an_correct_means.append(an_corrects.mean().item() if len(an_corrects) > 0 else 0)
        an_incorrect_means.append(an_incorrects.mean().item() if len(an_incorrects) > 0 else 0)
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # Set up bar positions
    x_pos = np.arange(len(model_names))
    bar_width = 0.2
    
    # Create bars with different patterns and colors
    # Steel blue for correct, dark red for incorrect
    # Solid bars for 'a', dotted bars for 'an'
    bars1 = ax.bar(x_pos - 1.5*bar_width, a_correct_means, bar_width, 
                   color='steelblue', alpha=0.8, label='"a" Correct', 
                   edgecolor='black', linewidth=0.5)
    
    bars2 = ax.bar(x_pos - 0.5*bar_width, a_incorrect_means, bar_width,
                   color='darkred', alpha=0.8, label='"a" Incorrect',
                   edgecolor='black', linewidth=0.5)
    
    bars3 = ax.bar(x_pos + 0.5*bar_width, an_correct_means, bar_width,
                   color='steelblue', alpha=0.8, label='"an" Correct',
                   hatch='.', edgecolor='black', linewidth=0.5)
    
    bars4 = ax.bar(x_pos + 1.5*bar_width, an_incorrect_means, bar_width,
                   color='darkred', alpha=0.8, label='"an" Incorrect', 
                   hatch='.', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    #ax.set_xlabel('Model', fontsize=20)
    ax.set_ylabel('Prop. of Influence Through Selected Nodes', fontsize=20)
    ax.set_title('Mean Proportion of Influence Through Selected Nodes by Model', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper left', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('relevant_nodes_refined/influence.pdf')
    plt.show()

# Create the combined bar plot
create_combined_mean_influence_bar_plot(all_model_results, models)
# %%
model = 'qwen3-1.7b-relu-lowl0'

all_model_results[model]['metadata']['selected_influence'] = all_model_results[model]['per_example_cumsum_selected_influences'][:, -1].tolist()
all_model_results[model]['metadata']['selected_nodes_count'] = all_model_results[model]['extra_metadata']['selected_nodes_count']
# Get data for the scatter plot
metadata = all_model_results[model]['metadata']
x_data = metadata['selected_nodes_count']
y_data = metadata['selected_influence']
correctness = metadata['correct?']

# Create scatter plot with colors based on correctness
plt.figure(figsize=(10, 6))
correct_mask = correctness == True
incorrect_mask = correctness == False

plt.scatter(x_data[correct_mask], y_data[correct_mask], 
           color='green', alpha=0.6, s=50, label='Correct')
plt.scatter(x_data[incorrect_mask], y_data[incorrect_mask], 
           color='red', alpha=0.6, s=50, label='Incorrect')

plt.xlabel('Selected Nodes Count')
plt.ylabel('Selected Influence')
plt.title('Selected Nodes Count vs Selected Influence (Qwen3-14B)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%
