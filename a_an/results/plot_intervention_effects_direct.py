#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]
dfs = {model: pd.read_csv(f'interventions_direct_last/{model}.csv') for model in models}

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

results = calculate_p_correct_differences(dfs)
dfs_correct = {k: df[df['correct?']] for k, df in dfs.items()}
dfs_incorrect = {k: df[~df['correct?']] for k, df in dfs.items()}
results_correct = calculate_p_correct_differences({k:df[df['correct?']] for k, df in dfs.items()})
results_incorrect = calculate_p_correct_differences({k:df[~df['correct?']] for k, df in dfs.items()})
#%%
# Create scatter plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Colors for different models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
model_positions = {model: i for i, model in enumerate(models)}

# Plot 1: Zeroed intervention effects
for i, (model, data) in enumerate(results_correct.items()):
    x_positions = np.full(len(data['zeroed_diff']), i) + np.random.normal(0, 0.05, len(data['zeroed_diff']))
    ax1.scatter(x_positions, data['zeroed_diff'], alpha=0.6, color=colors[i], label=model, s=30)

ax1.set_xlabel('Model')
ax1.set_ylabel('Change in p(correct)')
ax1.set_title('Effect of Zero Ablations on p(correct)')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Multiplied intervention effects
for i, (model, data) in enumerate(results_incorrect.items()):
    x_positions = np.full(len(data['multiplied_diff']), i) + np.random.normal(0, 0.05, len(data['multiplied_diff']))
    ax2.scatter(x_positions, data['multiplied_diff'], alpha=0.6, color=colors[i], s=30)

ax2.set_xlabel('Model')
ax2.set_ylabel('Change in p(correct)')
ax2.set_title('Effect of Multiplying Interventions on p(correct)')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

# Add shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(models))

plt.tight_layout()
plt.show()
#%%

# Create vertically stacked plots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Colors for different models
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
model_positions = {model: i for i, model in enumerate(models)}

# Plot 1: Zeroed intervention effects
for i, (model, data) in enumerate(results_correct.items()):
    x_positions = np.full(len(data['zeroed_diff']), i) + np.random.normal(0, 0.05, len(data['zeroed_diff']))
    ax1.scatter(x_positions, data['zeroed_diff'], alpha=0.6, color=colors[i], label=model, s=30)

ax1.set_ylabel('Change in p(correct)')
ax1.set_title('Effect of Zero Ablations on p(correct)')
ax1.set_xticks(range(len(models)))
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Multiplied intervention effects
for i, (model, data) in enumerate(results_incorrect.items()):
    x_positions = np.full(len(data['multiplied_diff']), i) + np.random.normal(0, 0.05, len(data['multiplied_diff']))
    ax2.scatter(x_positions, data['multiplied_diff'], alpha=0.6, color=colors[i], s=30)

ax2.set_xlabel('Model')
ax2.set_ylabel('Change in p(correct)')
ax2.set_title('Effect of Multiplying Interventions on p(correct)')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

# Add shared legend
handles, labels = ax1.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.035), ncol=len(models))

plt.tight_layout()
plt.savefig('interventions_direct_last/intervention_effects.pdf', bbox_inches='tight')
plt.show()
#%%
# Create color variants for a/an articles
def create_article_color_variants(base_color):
    # Convert hex to RGB
    base_color = base_color.lstrip('#')
    r, g, b = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Create darker variant for "a" articles (multiply by 0.8)
    a_color = f'#{int(r*0.8):02x}{int(g*0.8):02x}{int(b*0.8):02x}'
    
    # Create lighter variant for "an" articles (interpolate with white)
    an_color = f'#{int(r + (255-r)*0.4):02x}{int(g + (255-g)*0.4):02x}{int(b + (255-b)*0.4):02x}'
    
    return a_color, an_color

# Create vertically stacked plots with shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Base colors for different models
base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot 1: Zeroed intervention effects
for i, (model, data) in enumerate(results_correct.items()):
    
    # Separate "a" and "an" examples
    a_article_mask = dfs_correct[model]['correct_articles'] == 'a'
    an_article_mask = dfs_correct[model]['correct_articles'] == 'an'
    
    # Plot "a" examples with full circles
    if np.any(a_article_mask):
        x_positions_a = np.full(np.sum(a_article_mask), i) + np.random.normal(0, 0.05, np.sum(a_article_mask))
        ax1.scatter(x_positions_a, data['zeroed_diff'][a_article_mask.values], 
                   alpha=0.7, color=base_colors[i], s=40, marker='o',
                   label=f'{model}' if i == 0 else "")
    
    # Plot "an" examples with empty stars
    if np.any(an_article_mask):
        x_positions_an = np.full(np.sum(an_article_mask), i) + np.random.normal(0, 0.05, np.sum(an_article_mask))
        ax1.scatter(x_positions_an, data['zeroed_diff'][an_article_mask.values], 
                   alpha=0.7, color=base_colors[i], s=60, marker='*', 
                   facecolors='none', edgecolors=base_colors[i], linewidths=1.5)

ax1.set_ylabel('Change in p(correct)')
ax1.set_title('Effect of Zero Ablations on p(correct)')
ax1.set_xticks(range(len(models)))
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Multiplied intervention effects
for i, (model, data) in enumerate(results_incorrect.items()):
    
    # Separate "a" and "an" examples
    a_article_mask = dfs_incorrect[model]['correct_articles'] == 'a'
    an_article_mask = dfs_incorrect[model]['correct_articles'] == 'an'
    
    # Plot "a" examples with full circles
    if np.any(a_article_mask):
        x_positions_a = np.full(np.sum(a_article_mask), i) + np.random.normal(0, 0.05, np.sum(a_article_mask))
        ax2.scatter(x_positions_a, data['multiplied_diff'][a_article_mask.values], 
                   alpha=0.7, color=base_colors[i], s=40, marker='o')
    
    # Plot "an" examples with empty stars
    if np.any(an_article_mask):
        x_positions_an = np.full(np.sum(an_article_mask), i) + np.random.normal(0, 0.05, np.sum(an_article_mask))
        ax2.scatter(x_positions_an, data['multiplied_diff'][an_article_mask.values], 
                   alpha=0.7, color=base_colors[i], s=60, marker='*',
                   facecolors='none', edgecolors=base_colors[i], linewidths=1.5)

ax2.set_xlabel('Model')
ax2.set_ylabel('Change in p(correct)')
ax2.set_title('Effect of Multiplying Interventions on p(correct)')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

# Add article type legend with marker styles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', 
           markersize=8, label='correct article = "a"'),
    Line2D([0], [0], marker='*', color='black', linestyle='None', 
           markersize=10, markerfacecolor='none', markeredgecolor='black',
           markeredgewidth=1.5, label='correct article = "an"')
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.tight_layout()
plt.savefig('interventions_direct_last/intervention_effects.pdf', bbox_inches='tight')
plt.show()
# %%
