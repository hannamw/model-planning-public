#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "serif"


models = [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]
dfs = {model: pd.read_csv(f'{model}.csv') for model in models}

#%%
# Process data to calculate p(correct) differences for all three intervention types
def calculate_p_correct_differences_all_interventions(dfs):
    results = {}
    
    for model, df in dfs.items():
        # Calculate original p(correct) and intervention effects
        original_p_correct = []
        
        # Normal intervention (all_*)
        all_zeroed_p_correct = []
        all_multiplied_p_correct = []
        
        # Direct intervention (direct_*)
        direct_zeroed_p_correct = []
        direct_multiplied_p_correct = []
        
        # Random intervention (random_*)
        random_zeroed_p_correct = []
        random_multiplied_p_correct = []
        
        for _, row in df.iterrows():
            # The 'article' column contains the correct article
            correct_article = row['article']
            
            # Original probabilities
            if correct_article == 'a':
                orig_correct = row['original_a_prob']
                all_zeroed_correct = row['all_zeroed_a_prob']
                all_multiplied_correct = row['all_multiplied_a_prob']
                direct_zeroed_correct = row['direct_zeroed_a_prob']
                direct_multiplied_correct = row['direct_multiplied_a_prob']
                random_zeroed_correct = row['random_zeroed_a_prob']
                random_multiplied_correct = row['random_multiplied_a_prob']
            else:  # correct_article == 'an'
                orig_correct = row['original_an_prob']
                all_zeroed_correct = row['all_zeroed_an_prob']
                all_multiplied_correct = row['all_multiplied_an_prob']
                direct_zeroed_correct = row['direct_zeroed_an_prob']
                direct_multiplied_correct = row['direct_multiplied_an_prob']
                random_zeroed_correct = row['random_zeroed_an_prob']
                random_multiplied_correct = row['random_multiplied_an_prob']
            
            original_p_correct.append(orig_correct)
            all_zeroed_p_correct.append(all_zeroed_correct)
            all_multiplied_p_correct.append(all_multiplied_correct)
            direct_zeroed_p_correct.append(direct_zeroed_correct)
            direct_multiplied_p_correct.append(direct_multiplied_correct)
            random_zeroed_p_correct.append(random_zeroed_correct)
            random_multiplied_p_correct.append(random_multiplied_correct)
        
        # Calculate differences
        original_p_correct = np.array(original_p_correct)
        
        results[model] = {
            'all_zeroed_diff': np.array(all_zeroed_p_correct) - original_p_correct,
            'all_multiplied_diff': np.array(all_multiplied_p_correct) - original_p_correct,
            'direct_zeroed_diff': np.array(direct_zeroed_p_correct) - original_p_correct,
            'direct_multiplied_diff': np.array(direct_multiplied_p_correct) - original_p_correct,
            'random_zeroed_diff': np.array(random_zeroed_p_correct) - original_p_correct,
            'random_multiplied_diff': np.array(random_multiplied_p_correct) - original_p_correct,
            'original_p_correct': original_p_correct
        }
    
    return results

results = calculate_p_correct_differences_all_interventions(dfs)

# Filter for correct and incorrect predictions
dfs_correct = {k: df[df['article_correct']] for k, df in dfs.items()}
dfs_incorrect = {k: df[~df['article_correct']] for k, df in dfs.items()}
results_correct = calculate_p_correct_differences_all_interventions(dfs_correct)
results_incorrect = calculate_p_correct_differences_all_interventions(dfs_incorrect)

#%%
# Calculate mean effects for line plots
def calculate_mean_effects_all_interventions(results_dict, dfs_dict):
    mean_effects = {}
    for model in models:
        data = results_dict[model]
        df = dfs_dict[model]
        
        # The 'article' column contains the correct articles
        correct_articles = df['article'].values
        a_article_mask = correct_articles == 'a'
        an_article_mask = correct_articles == 'an'
        
        mean_effects[model] = {}
        
        # Calculate means for each intervention type and operation
        for intervention in ['all', 'direct', 'random']:
            for operation in ['zeroed', 'multiplied']:
                key = f'{intervention}_{operation}_diff'
                mean_effects[model][f'{intervention}_{operation}_a_mean'] = (
                    np.mean(data[key][a_article_mask]) if np.any(a_article_mask) else np.nan
                )
                mean_effects[model][f'{intervention}_{operation}_an_mean'] = (
                    np.mean(data[key][an_article_mask]) if np.any(an_article_mask) else np.nan
                )
    
    return mean_effects

mean_effects_correct = calculate_mean_effects_all_interventions(results_correct, dfs_correct)
mean_effects_incorrect = calculate_mean_effects_all_interventions(results_incorrect, dfs_incorrect)

#%%
# Create three separate plots for each intervention type (following original script style)
intervention_types = ['all', 'direct', 'random']
intervention_labels = ['Normal Intervention', 'Direct Intervention', 'Random Intervention']

# Colors for different models (from original script)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
model_positions = {model: i for i, model in enumerate(models)}

# Base colors for different article types (for line plots)
a_color = '#2E86AB'  # Blue for "a"
an_color = '#A23B72'  # Purple for "an"

from matplotlib.lines import Line2D

for intervention, label in zip(intervention_types, intervention_labels):
    # Create vertically stacked plots with shared x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    
    # Plot 1: Zeroed intervention effects (correct predictions) - SCATTER POINTS
    for i, (model, data) in enumerate(results_correct.items()):
        df_model = dfs_correct[model]
        x_positions_scatter = np.full(len(data[f'{intervention}_zeroed_diff']), i) + np.random.normal(0, 0.05, len(data[f'{intervention}_zeroed_diff']))
        
        # Separate by article type for different markers
        article_mask_a = df_model['article'] == 'a'
        article_mask_an = df_model['article'] == 'an'
        
        if np.any(article_mask_a):
            ax1.scatter(x_positions_scatter[article_mask_a], data[f'{intervention}_zeroed_diff'][article_mask_a], 
                       alpha=0.6, color=a_color, marker='o', s=30)
        if np.any(article_mask_an):
            ax1.scatter(x_positions_scatter[article_mask_an], data[f'{intervention}_zeroed_diff'][article_mask_an], 
                       alpha=0.6, color=an_color, marker='*', s=50)
    
    # Add line plot overlay
    x_positions = range(len(models))
    zeroed_a_means = [mean_effects_correct[model][f'{intervention}_zeroed_a_mean'] for model in models]
    zeroed_an_means = [mean_effects_correct[model][f'{intervention}_zeroed_an_mean'] for model in models]
    
    ax1.plot(x_positions, zeroed_a_means, color=a_color, marker='o', linewidth=2, 
             markersize=6, alpha=0.8, zorder=10)
    ax1.plot(x_positions, zeroed_an_means, color=an_color, marker='*', linewidth=2, 
             markersize=8, alpha=0.8, zorder=10)
    
    ax1.set_ylabel('Change in p(correct)')
    ax1.set_title(f'{label}\nEffect of Zero Ablations on p(correct)')
    ax1.set_xticks(range(len(models)))
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multiplied intervention effects (incorrect predictions) - SCATTER POINTS
    for i, (model, data) in enumerate(results_incorrect.items()):
        df_model = dfs_incorrect[model]
        x_positions_scatter = np.full(len(data[f'{intervention}_multiplied_diff']), i) + np.random.normal(0, 0.05, len(data[f'{intervention}_multiplied_diff']))
        
        # Separate by article type for different markers
        article_mask_a = df_model['article'] == 'a'
        article_mask_an = df_model['article'] == 'an'
        
        if np.any(article_mask_a):
            ax2.scatter(x_positions_scatter[article_mask_a], data[f'{intervention}_multiplied_diff'][article_mask_a], 
                       alpha=0.6, color=a_color, marker='o', s=30)
        if np.any(article_mask_an):
            ax2.scatter(x_positions_scatter[article_mask_an], data[f'{intervention}_multiplied_diff'][article_mask_an], 
                       alpha=0.6, color=an_color, marker='*', s=50)
    
    # Add line plot overlay
    multiplied_a_means = [mean_effects_incorrect[model][f'{intervention}_multiplied_a_mean'] for model in models]
    multiplied_an_means = [mean_effects_incorrect[model][f'{intervention}_multiplied_an_mean'] for model in models]
    
    ax2.plot(x_positions, multiplied_a_means, color=a_color, marker='o', linewidth=2, 
             markersize=6, alpha=0.8, zorder=10)
    ax2.plot(x_positions, multiplied_an_means, color=an_color, marker='*', linewidth=2, 
             markersize=8, alpha=0.8, zorder=10)
    
    ax2.set_ylabel('Change in p(correct)')
    ax2.set_title('Effect of Multiplying Interventions on p(correct)')
    ax2.set_xlabel('Model')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=0)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Create simple 2-item legend (removing trendlines from legend)
    legend_elements = [
        Line2D([0], [0], color=a_color, marker='o', linewidth=0, markersize=6,
               label='individual "a" points', alpha=0.6),
        Line2D([0], [0], color=an_color, marker='*', linewidth=0, markersize=8,
               label='individual "an" points', alpha=0.6)
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    
    plt.tight_layout()
    plt.savefig(f'a_an_{intervention}_intervention_effects.pdf', bbox_inches='tight')
    plt.show()

#%%
# Create combined plot with all effects and direct effects side by side
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(12, 5.5), sharex=True)

# Left column: All effects (Normal Intervention)
intervention = 'all'

# Plot 1: Zeroed intervention effects (correct predictions) - SCATTER POINTS
for i, (model, data) in enumerate(results_correct.items()):
    df_model = dfs_correct[model]
    x_positions_scatter = np.full(len(data[f'{intervention}_zeroed_diff']), i) + np.random.normal(0, 0.05, len(data[f'{intervention}_zeroed_diff']))
    
    # Separate by article type for different markers
    article_mask_a = df_model['article'] == 'a'
    article_mask_an = df_model['article'] == 'an'
    
    if np.any(article_mask_a):
        ax1.scatter(x_positions_scatter[article_mask_a], data[f'{intervention}_zeroed_diff'][article_mask_a], 
                   alpha=0.6, color=a_color, marker='o', s=30)
    if np.any(article_mask_an):
        ax1.scatter(x_positions_scatter[article_mask_an], data[f'{intervention}_zeroed_diff'][article_mask_an], 
                   alpha=0.6, color=an_color, marker='*', s=50)

# Add line plot overlay
x_positions = range(len(models))
zeroed_a_means = [mean_effects_correct[model][f'{intervention}_zeroed_a_mean'] for model in models]
zeroed_an_means = [mean_effects_correct[model][f'{intervention}_zeroed_an_mean'] for model in models]

ax1.plot(x_positions, zeroed_a_means, color=a_color, marker='o', linewidth=2, 
         markersize=6, alpha=0.8, zorder=10)
ax1.plot(x_positions, zeroed_an_means, color=an_color, marker='*', linewidth=2, 
         markersize=8, alpha=0.8, zorder=10)

ax1.set_ylabel('Change in p(correct)')
ax1.set_title('Effect of Zero Ablations on p(correct)')
ax1.set_xticks(range(len(models)))
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.grid(True, alpha=0.3)

# Plot 2: Multiplied intervention effects (incorrect predictions) - SCATTER POINTS
for i, (model, data) in enumerate(results_incorrect.items()):
    df_model = dfs_incorrect[model]
    x_positions_scatter = np.full(len(data[f'{intervention}_multiplied_diff']), i) + np.random.normal(0, 0.05, len(data[f'{intervention}_multiplied_diff']))
    
    # Separate by article type for different markers
    article_mask_a = df_model['article'] == 'a'
    article_mask_an = df_model['article'] == 'an'
    
    if np.any(article_mask_a):
        ax2.scatter(x_positions_scatter[article_mask_a], data[f'{intervention}_multiplied_diff'][article_mask_a], 
                   alpha=0.6, color=a_color, marker='o', s=30)
    if np.any(article_mask_an):
        ax2.scatter(x_positions_scatter[article_mask_an], data[f'{intervention}_multiplied_diff'][article_mask_an], 
                   alpha=0.6, color=an_color, marker='*', s=50)

# Add line plot overlay
multiplied_a_means = [mean_effects_incorrect[model][f'{intervention}_multiplied_a_mean'] for model in models]
multiplied_an_means = [mean_effects_incorrect[model][f'{intervention}_multiplied_an_mean'] for model in models]

ax2.plot(x_positions, multiplied_a_means, color=a_color, marker='o', linewidth=2, 
         markersize=6, alpha=0.8, zorder=10)
ax2.plot(x_positions, multiplied_an_means, color=an_color, marker='*', linewidth=2, 
         markersize=8, alpha=0.8, zorder=10)

ax2.set_ylabel('Change in p(correct)')
ax2.set_title('Effect of Multiplying Interventions on p(correct)')
ax2.set_xlabel('Model')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=0)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.grid(True, alpha=0.3)

# Right column: Direct effects
intervention = 'direct'

# Plot 3: Zeroed intervention effects (correct predictions) - SCATTER POINTS
for i, (model, data) in enumerate(results_correct.items()):
    df_model = dfs_correct[model]
    x_positions_scatter = np.full(len(data[f'{intervention}_zeroed_diff']), i) + np.random.normal(0, 0.05, len(data[f'{intervention}_zeroed_diff']))
    
    # Separate by article type for different markers
    article_mask_a = df_model['article'] == 'a'
    article_mask_an = df_model['article'] == 'an'
    
    if np.any(article_mask_a):
        ax3.scatter(x_positions_scatter[article_mask_a], data[f'{intervention}_zeroed_diff'][article_mask_a], 
                   alpha=0.6, color=a_color, marker='o', s=30)
    if np.any(article_mask_an):
        ax3.scatter(x_positions_scatter[article_mask_an], data[f'{intervention}_zeroed_diff'][article_mask_an], 
                   alpha=0.6, color=an_color, marker='*', s=50)

# Add line plot overlay
zeroed_a_means = [mean_effects_correct[model][f'{intervention}_zeroed_a_mean'] for model in models]
zeroed_an_means = [mean_effects_correct[model][f'{intervention}_zeroed_an_mean'] for model in models]

ax3.plot(x_positions, zeroed_a_means, color=a_color, marker='o', linewidth=2, 
         markersize=6, alpha=0.8, zorder=10)
ax3.plot(x_positions, zeroed_an_means, color=an_color, marker='*', linewidth=2, 
         markersize=8, alpha=0.8, zorder=10)

ax3.set_title('Effect of Direct Zero Ablations on p(correct)')
ax3.set_xticks(range(len(models)))
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(True, alpha=0.3)

# Plot 4: Multiplied intervention effects (incorrect predictions) - SCATTER POINTS
for i, (model, data) in enumerate(results_incorrect.items()):
    df_model = dfs_incorrect[model]
    x_positions_scatter = np.full(len(data[f'{intervention}_multiplied_diff']), i) + np.random.normal(0, 0.05, len(data[f'{intervention}_multiplied_diff']))
    
    # Separate by article type for different markers
    article_mask_a = df_model['article'] == 'a'
    article_mask_an = df_model['article'] == 'an'
    
    if np.any(article_mask_a):
        ax4.scatter(x_positions_scatter[article_mask_a], data[f'{intervention}_multiplied_diff'][article_mask_a], 
                   alpha=0.6, color=a_color, marker='o', s=30)
    if np.any(article_mask_an):
        ax4.scatter(x_positions_scatter[article_mask_an], data[f'{intervention}_multiplied_diff'][article_mask_an], 
                   alpha=0.6, color=an_color, marker='*', s=50)

# Add line plot overlay
multiplied_a_means = [mean_effects_incorrect[model][f'{intervention}_multiplied_a_mean'] for model in models]
multiplied_an_means = [mean_effects_incorrect[model][f'{intervention}_multiplied_an_mean'] for model in models]

ax4.plot(x_positions, multiplied_a_means, color=a_color, marker='o', linewidth=2, 
         markersize=6, alpha=0.8, zorder=10)
ax4.plot(x_positions, multiplied_an_means, color=an_color, marker='*', linewidth=2, 
         markersize=8, alpha=0.8, zorder=10)

ax4.set_title('Effect of Direct Multiplying Interventions on p(correct)')
ax4.set_xlabel('Model')
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(models, rotation=0)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.grid(True, alpha=0.3)

# Create shared legend
legend_elements = [
    Line2D([0], [0], color=a_color, marker='o', linewidth=2, markersize=6, 
           label='correct article = "a" (trend)', alpha=0.8),
    Line2D([0], [0], color=an_color, marker='*', linewidth=2, markersize=8,
           label='correct article = "an" (trend)', alpha=0.8),
    Line2D([0], [0], color=a_color, marker='o', linewidth=0, markersize=6,
           label='individual "a" points', alpha=0.6),
    Line2D([0], [0], color=an_color, marker='*', linewidth=0, markersize=8,
           label='individual "an" points', alpha=0.6)
]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=4)

plt.tight_layout()
plt.savefig('combined_all_direct_intervention_effects.pdf', bbox_inches='tight')
plt.show()

#%%
