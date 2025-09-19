#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

models = [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]

# Load data from both directories
dfs_last = {model: pd.read_csv(f'../interventions_last/{model}.csv') for model in models}
dfs_direct = {model: pd.read_csv(f'../interventions_direct_last/{model}.csv') for model in models}

#%%
# Process data to calculate p(correct) differences
def calculate_p_correct_differences(dfs):
    results = {}
    
    for model, df in dfs.items():
        # Calculate original p(correct)
        original_p_correct = []
        zeroed_p_correct = []
        multiplied_p_correct = []
        
        for _, row in df.iterrows():
            correct_article = row['correct_articles']
            
            # Original probabilities
            if correct_article == 'a':
                orig_correct = row['p(a)']
                zeroed_correct = row['zeroed_a_prob']
                multiplied_correct = row['multiplied_a_prob']
            else:  # correct_article == 'an'
                orig_correct = row['p(an)']
                zeroed_correct = row['zeroed_an_prob']
                multiplied_correct = row['multiplied_an_prob']
            
            original_p_correct.append(orig_correct)
            zeroed_p_correct.append(zeroed_correct)
            multiplied_p_correct.append(multiplied_correct)
        
        # Calculate differences
        zeroed_diff = np.array(zeroed_p_correct) - np.array(original_p_correct)
        multiplied_diff = np.array(multiplied_p_correct) - np.array(original_p_correct)
        
        results[model] = {
            'zeroed_diff': zeroed_diff,
            'multiplied_diff': multiplied_diff,
            'original_p_correct': np.array(original_p_correct)
        }
    
    return results

# Process both datasets
results_last = calculate_p_correct_differences(dfs_last)
results_direct = calculate_p_correct_differences(dfs_direct)

# Split by correct/incorrect predictions
dfs_last_correct = {k: df[df['correct?']] for k, df in dfs_last.items()}
dfs_last_incorrect = {k: df[~df['correct?']] for k, df in dfs_last.items()}
dfs_direct_correct = {k: df[df['correct?']] for k, df in dfs_direct.items()}
dfs_direct_incorrect = {k: df[~df['correct?']] for k, df in dfs_direct.items()}

results_last_correct = calculate_p_correct_differences(dfs_last_correct)
results_last_incorrect = calculate_p_correct_differences(dfs_last_incorrect)
results_direct_correct = calculate_p_correct_differences(dfs_direct_correct)
results_direct_incorrect = calculate_p_correct_differences(dfs_direct_incorrect)

#%%
# Create 2x2 combined plot with shared x-axis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

# Base colors for different models
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

def plot_intervention_effects(ax, results_data, dfs_data, intervention_type, title):
    """Helper function to plot intervention effects"""
    for i, (model, data) in enumerate(results_data.items()):
        # Separate "a" and "an" examples
        a_article_mask = dfs_data[model]['correct_articles'] == 'a'
        an_article_mask = dfs_data[model]['correct_articles'] == 'an'
        
        # Choose which difference to plot
        diff_data = data['zeroed_diff'] if intervention_type == 'zeroed' else data['multiplied_diff']
        
        # Plot "a" examples with full circles
        if np.any(a_article_mask):
            x_positions_a = np.full(np.sum(a_article_mask), i) + np.random.normal(0, 0.05, np.sum(a_article_mask))
            ax.scatter(x_positions_a, diff_data[a_article_mask.values], 
                      alpha=0.7, color=base_colors[i], s=40, marker='o')
        
        # Plot "an" examples with empty stars
        if np.any(an_article_mask):
            x_positions_an = np.full(np.sum(an_article_mask), i) + np.random.normal(0, 0.05, np.sum(an_article_mask))
            ax.scatter(x_positions_an, diff_data[an_article_mask.values], 
                      alpha=0.7, color=base_colors[i], s=60, marker='*',
                      facecolors='none', edgecolors=base_colors[i], linewidths=1.5)
    
    ax.set_ylabel('Change in p(correct)')
    ax.set_title(title)
    ax.set_xticks(range(len(models)))
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

# Top left: Zero ablations on correct predictions (interventions_last)
plot_intervention_effects(ax1, results_last_correct, dfs_last_correct, 'zeroed', 
                         'Zero Ablations on p(correct)')

# Top right: Zero ablations on correct predictions (interventions_direct_last)
plot_intervention_effects(ax2, results_direct_correct, dfs_direct_correct, 'zeroed', 
                         'Direct Zero Ablations on p(correct)')

# Bottom left: Multiplying interventions on incorrect predictions (interventions_last)
plot_intervention_effects(ax3, results_last_incorrect, dfs_last_incorrect, 'multiplied', 
                         'Multiplying Interventions on p(correct)')
ax3.set_xticklabels(models, rotation=0)

# Bottom right: Multiplying interventions on incorrect predictions (interventions_direct_last)
plot_intervention_effects(ax4, results_direct_incorrect, dfs_direct_incorrect, 'multiplied', 
                         'Direct Multiplying Interventions on p(correct)')
ax4.set_xticklabels(models, rotation=0)

# Add shared legend at bottom center
legend_elements = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', 
           markersize=8, label='correct article = "a"'),
    Line2D([0], [0], marker='*', color='black', linestyle='None', 
           markersize=10, markerfacecolor='none', markeredgecolor='black',
           markeredgewidth=1.5, label='correct article = "an"')
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2)

plt.tight_layout()
plt.savefig('combined_intervention_effects.pdf', bbox_inches='tight')
plt.show()
# %%
