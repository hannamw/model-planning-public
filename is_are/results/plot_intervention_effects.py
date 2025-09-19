#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]
dfs = {model: pd.read_csv(f'interventions_last/{model}.csv') for model in models}
for df in dfs.values():
    df['correct?'] = [(is_prob > are_prob) if answer == 'is' else (are_prob > is_prob) for answer, is_prob, are_prob in zip(df['answer'], df['original_is_prob'], df['original_are_prob'])]
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
            correct_article = row['answer']
            
            # Original probabilities
            if correct_article == 'are':
                orig_correct = row['original_are_prob']
                zeroed_correct = row['zeroed_are_prob']
                multiplied_correct = row['multiplied_are_prob']
            else:  # correct_article == 'an'
                orig_correct = row['original_is_prob']
                zeroed_correct = row['zeroed_is_prob']
                multiplied_correct = row['multiplied_is_prob']
            
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
plt.savefig('interventions_last/intervention_effects.pdf', bbox_inches='tight')
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
    
    # Separate "are" and "is" examples
    are_mask = dfs_correct[model]['answer'] == 'are'
    is_mask = dfs_correct[model]['answer'] == 'is'
    
    # Plot "are" examples with full circles
    if np.any(are_mask):
        x_positions_are = np.full(np.sum(are_mask), i) + np.random.normal(0, 0.05, np.sum(are_mask))
        ax1.scatter(x_positions_are, data['zeroed_diff'][are_mask.values], 
                   alpha=0.7, color=base_colors[i], s=40, marker='o',
                   label=f'{model}' if i == 0 else "")
    
    # Plot "is" examples with empty stars
    if np.any(is_mask):
        x_positions_is = np.full(np.sum(is_mask), i) + np.random.normal(0, 0.05, np.sum(is_mask))
        ax1.scatter(x_positions_is, data['zeroed_diff'][is_mask.values], 
                   alpha=0.7, color=base_colors[i], s=60, marker='*', 
                   facecolors='none', edgecolors=base_colors[i], linewidths=1.5)

ax1.set_ylabel('Change in p(correct)')
ax1.set_title('Effect of Zero Ablations on p(correct)')
ax1.set_xticks(range(len(models)))
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Multiplied intervention effects
for i, (model, data) in enumerate(results_incorrect.items()):
    
    # Separate "are" and "is" examples
    are_mask = dfs_incorrect[model]['answer'] == 'are'
    is_mask = dfs_incorrect[model]['answer'] == 'is'
    
    # Plot "are" examples with full circles
    if np.any(are_mask):
        x_positions_are = np.full(np.sum(are_mask), i) + np.random.normal(0, 0.05, np.sum(are_mask))
        ax2.scatter(x_positions_are, data['multiplied_diff'][are_mask.values], 
                   alpha=0.7, color=base_colors[i], s=40, marker='o')
    
    # Plot "is" examples with empty stars
    if np.any(is_mask):
        x_positions_is = np.full(np.sum(is_mask), i) + np.random.normal(0, 0.05, np.sum(is_mask))
        ax2.scatter(x_positions_is, data['multiplied_diff'][is_mask.values], 
                   alpha=0.7, color=base_colors[i], s=60, marker='*',
                   facecolors='none', edgecolors=base_colors[i], linewidths=1.5)

ax2.set_xlabel('Model')
ax2.set_ylabel('Change in p(correct)')
ax2.set_title('Effect of Multiplying Interventions on p(correct)')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

# Add verb type legend with marker styles
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', linestyle='None', 
           markersize=8, label='correct verb = "are"'),
    Line2D([0], [0], marker='*', color='black', linestyle='None', 
           markersize=10, markerfacecolor='none', markeredgecolor='black',
           markeredgewidth=1.5, label='correct verb = "is"')
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)

plt.tight_layout()
plt.savefig('interventions_last/intervention_effects_is_are.pdf', bbox_inches='tight')
plt.show()
# %%
