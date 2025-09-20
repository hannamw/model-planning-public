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

# Calculate mean effects for line plots
def calculate_mean_effects(results_dict, dfs_dict):
    mean_effects = {}
    for model in models:
        data = results_dict[model]
        df = dfs_dict[model]
        
        # Separate "a" and "an" examples
        a_article_mask = df['correct_articles'] == 'a'
        an_article_mask = df['correct_articles'] == 'an'
        
        mean_effects[model] = {
            'zeroed_a_mean': np.mean(data['zeroed_diff'][a_article_mask.values]) if np.any(a_article_mask) else np.nan,
            'zeroed_an_mean': np.mean(data['zeroed_diff'][an_article_mask.values]) if np.any(an_article_mask) else np.nan,
            'multiplied_a_mean': np.mean(data['multiplied_diff'][a_article_mask.values]) if np.any(a_article_mask) else np.nan,
            'multiplied_an_mean': np.mean(data['multiplied_diff'][an_article_mask.values]) if np.any(an_article_mask) else np.nan,
        }
    return mean_effects

# Calculate mean effects for all datasets
mean_effects_last_correct = calculate_mean_effects(results_last_correct, dfs_last_correct)
mean_effects_last_incorrect = calculate_mean_effects(results_last_incorrect, dfs_last_incorrect)
mean_effects_direct_correct = calculate_mean_effects(results_direct_correct, dfs_direct_correct)
mean_effects_direct_incorrect = calculate_mean_effects(results_direct_incorrect, dfs_direct_incorrect)

#%%
# Create 2x2 combined plot with shared x-axis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

# Base colors for different article types
a_color = '#2E86AB'  # Blue for "a"
an_color = '#A23B72'  # Purple for "an"

# Prepare data for line plots
x_positions = range(len(models))

def plot_intervention_effects_line(ax, mean_effects_data, intervention_type, title):
    """Helper function to plot intervention effects as lines"""
    if intervention_type == 'zeroed':
        a_means = [mean_effects_data[model]['zeroed_a_mean'] for model in models]
        an_means = [mean_effects_data[model]['zeroed_an_mean'] for model in models]
    else:  # multiplied
        a_means = [mean_effects_data[model]['multiplied_a_mean'] for model in models]
        an_means = [mean_effects_data[model]['multiplied_an_mean'] for model in models]
    
    # Plot lines
    ax.plot(x_positions, a_means, color=a_color, marker='o', linewidth=2, 
            markersize=6, label='correct article = "a"')
    ax.plot(x_positions, an_means, color=an_color, marker='*', linewidth=2, 
            markersize=8, label='correct article = "an"')
    
    ax.set_ylabel('Change in p(correct)')
    ax.set_title(title)
    ax.set_xticks(range(len(models)))
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

# Top left: Zero ablations on correct predictions (interventions_last)
plot_intervention_effects_line(ax1, mean_effects_last_correct, 'zeroed', 
                              'Zero Ablations on p(correct)')

# Top right: Zero ablations on correct predictions (interventions_direct_last)
plot_intervention_effects_line(ax2, mean_effects_direct_correct, 'zeroed', 
                              'Direct Zero Ablations on p(correct)')

# Bottom left: Multiplying interventions on incorrect predictions (interventions_last)
plot_intervention_effects_line(ax3, mean_effects_last_incorrect, 'multiplied', 
                              'Multiplying Interventions on p(correct)')
ax3.set_xticklabels(models, rotation=0)

# Bottom right: Multiplying interventions on incorrect predictions (interventions_direct_last)
plot_intervention_effects_line(ax4, mean_effects_direct_incorrect, 'multiplied', 
                              'Direct Multiplying Interventions on p(correct)')
ax4.set_xticklabels(models, rotation=0)

# Add shared legend at bottom center with line styles
legend_elements = [
    Line2D([0], [0], color=a_color, marker='o', linewidth=2, markersize=6, 
           label='correct article = "a"'),
    Line2D([0], [0], color=an_color, marker='*', linewidth=2, markersize=8,
           label='correct article = "an"')
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2)

plt.tight_layout()
plt.savefig('combined_intervention_effects_line.pdf', bbox_inches='tight')
plt.show()
# %%
