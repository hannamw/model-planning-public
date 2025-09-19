#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]

# Create figure with 3x2 subplots with shared x-axis
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

# Flatten axes for easier indexing
axes_flat = axes.flatten()
subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Plot A: EOL Stopped Length Diffs
dfs_eol = {model: pd.read_csv(f'eol_intervention/{model}.csv') for model in models}
y_values = [dfs_eol[model]['eol_stopped_len_diffs'].mean() for model in models]
axes_flat[0].plot(range(len(models)), y_values, marker='o', linewidth=2, markersize=8)
axes_flat[0].set_xticks(range(len(models)))
axes_flat[0].set_xticklabels(models)
axes_flat[0].set_title('Effect of upweighting EOL on line length')
axes_flat[0].text(0.08, 0.5, 'A', transform=axes_flat[0].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[0].set_ylabel('Change in line length (tokens)')
axes_flat[0].grid(True, alpha=0.3)

# Plot B: EOL Continued Length Diffs
y_values = [dfs_eol[model]['eol_continued_len_diffs'].mean() for model in models]
axes_flat[1].plot(range(len(models)), y_values, marker='o', linewidth=2, markersize=8)
axes_flat[1].set_xticks(range(len(models)))
axes_flat[1].set_xticklabels(models)
axes_flat[1].set_title('Effect of downweighting EOL on line length')
axes_flat[1].text(0.92, 0.5, 'B', transform=axes_flat[1].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[1].set_ylabel('Change in line length (tokens)')
axes_flat[1].grid(True, alpha=0.3)

# Plot C: EOL Intervention Rhyme - Both Metrics
dfs_eol_rhyme = {model: pd.read_csv(f'eol_intervention_rhyme/{model}.csv') for model in models}
vowel_and_consonant_correct = [(dfs_eol_rhyme[model]['vowel_correct'] & dfs_eol_rhyme[model]['consonant_correct']).mean() for model in models]
vowel_and_singular_consonant_correct = [(dfs_eol_rhyme[model]['vowel_correct'] & dfs_eol_rhyme[model]['singular_consonant_correct']).mean() for model in models]
x_pos = np.arange(len(models))
axes_flat[2].plot(x_pos, vowel_and_consonant_correct, marker='o', linewidth=2, markersize=8, label='Vowel & Consonant')
axes_flat[2].plot(x_pos, vowel_and_singular_consonant_correct, marker='s', linewidth=2, markersize=8, label='Vowel & Singular Consonant')
axes_flat[2].set_xticks(x_pos)
axes_flat[2].set_xticklabels(models)
axes_flat[2].set_title('Effect of downweighting EOL on rhyming accuracy')
axes_flat[2].text(0.08, 0.5, 'C', transform=axes_flat[2].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[2].set_ylabel('Rhyming Accuracy')
axes_flat[2].legend()
axes_flat[2].grid(True, alpha=0.3)

# Plot D: NEOL Stopped Length Diffs
dfs_neol = {model: pd.read_csv(f'neol_intervention/{model}.csv') for model in models}
y_values = [dfs_neol[model]['neol_stopped_len_diffs'].mean() for model in models]
axes_flat[3].plot(range(len(models)), y_values, marker='o', linewidth=2, markersize=8)
axes_flat[3].set_xticks(range(len(models)))
axes_flat[3].set_xticklabels(models)
axes_flat[3].set_title('Effect of Upweighting NEOL on Line Length')
axes_flat[3].text(0.92, 0.5, 'D', transform=axes_flat[3].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[3].set_ylabel('Change in line length (tokens)')
axes_flat[3].grid(True, alpha=0.3)

# Plot E: NEOL Eventually Rhymed
y_values = [dfs_neol[model]['neol_eventually_rhymed'].mean() for model in models]
axes_flat[4].plot(range(len(models)), y_values, marker='o', linewidth=2, markersize=8)
axes_flat[4].set_xticks(range(len(models)))
axes_flat[4].set_xticklabels(models)
axes_flat[4].set_title('Effect of NEOL Ablation on Rhyming')
axes_flat[4].text(0.08, 0.5, 'E', transform=axes_flat[4].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[4].set_ylabel('Rhyming Accuracy')
axes_flat[4].grid(True, alpha=0.3)

# Plot F: Rhyme Intervention Sample NC - Both Metrics
dfs_rhyme_sample = {model: pd.read_csv(f'rhyme_intervention_sample_nc/{model}.csv') for model in models}
vowel_and_consonant_correct = [(dfs_rhyme_sample[model]['vowel_correct'] & dfs_rhyme_sample[model]['consonant_correct']).mean() for model in models]
vowel_and_singular_consonant_correct = [(dfs_rhyme_sample[model]['vowel_correct'] & dfs_rhyme_sample[model]['singular_consonant_correct']).mean() for model in models]
x_pos = np.arange(len(models))
axes_flat[5].plot(x_pos, vowel_and_consonant_correct, marker='o', linewidth=2, markersize=8, label='Vowel & Consonant')
axes_flat[5].plot(x_pos, vowel_and_singular_consonant_correct, marker='s', linewidth=2, markersize=8, label='Vowel & Singular Consonant')
axes_flat[5].set_xticks(x_pos)
axes_flat[5].set_xticklabels(models)
axes_flat[5].set_title('Effectiveness of Changing Rhyming Features')
axes_flat[5].text(0.92, 0.5, 'F', transform=axes_flat[5].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[5].set_ylabel('Rhyming Accuracy (for new rhyme)')
axes_flat[5].legend()
axes_flat[5].grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.show()
#%%