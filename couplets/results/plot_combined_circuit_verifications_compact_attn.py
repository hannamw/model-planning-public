#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

models = [f'Qwen3-{size}B' for size in [0.6, 1.7, 4, 8, 14]]

# Create figure with 2x2 subplots with shared x-axis (4 plots total after combining A&B and B&D)
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)

# Flatten axes for easier indexing
axes_flat = axes.flatten()
subplot_labels = ['A', 'B', 'C', 'D']

# Plot A: Combined EOL Effects (Upweighting and Downweighting)
dfs_eol = {model: pd.read_csv(f'eol_intervention/{model}.csv') for model in models}
y_values_stopped = [dfs_eol[model]['eol_stopped_len_diffs'].mean() for model in models]
y_values_continued = [dfs_eol[model]['eol_continued_len_diffs'].mean() for model in models]
x_pos = np.arange(len(models))
axes_flat[0].plot(x_pos, y_values_stopped, marker='o', linewidth=2, markersize=8, label='Upweighting EOL')
axes_flat[0].plot(x_pos, y_values_continued, marker='s', linewidth=2, markersize=8, label='Downweighting EOL')
axes_flat[0].set_xticks(x_pos)
axes_flat[0].set_xticklabels(models)
axes_flat[0].set_title('Effect of EOL interventions on line length')
axes_flat[0].text(0.92, 0.5, 'A', transform=axes_flat[0].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[0].set_ylabel('Change in line length (tokens)')
axes_flat[0].legend()
axes_flat[0].grid(True, alpha=0.3)

# Plot B: NEOL Stopped Length Diffs
dfs_neol = {model: pd.read_csv(f'neol_intervention/{model}.csv') for model in models}
y_values = [dfs_neol[model]['neol_stopped_len_diffs'].mean() for model in models]
axes_flat[1].plot(range(len(models)), y_values, marker='o', linewidth=2, markersize=8)
axes_flat[1].set_xticks(range(len(models)))
axes_flat[1].set_xticklabels(models)
axes_flat[1].set_title('Effect of Upweighting NEOL on Line Length')
axes_flat[1].text(0.92, 0.5, 'B', transform=axes_flat[1].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[1].set_ylabel('Change in line length (tokens)')
axes_flat[1].grid(True, alpha=0.3)

# Plot C: Combined Rhyming Effects (EOL Intervention and NEOL Eventually Rhymed)
dfs_eol_rhyme = {model: pd.read_csv(f'eol_intervention_rhyme/{model}.csv') for model in models}
dfs_neol_rhyme_attn = {model: pd.read_csv(f'attention_intervention/{model}.csv') for model in models}
vowel_and_consonant_correct = [(dfs_eol_rhyme[model]['vowel_correct'] & dfs_eol_rhyme[model]['consonant_correct']).mean() for model in models]
neol_eventually_rhymed = [dfs_neol[model]['neol_eventually_rhymed'].mean() for model in models]
attn_eventually_rhymed = [dfs_neol_rhyme_attn[model]['neol_eventually_rhymed'].mean() for model in models]
x_pos = np.arange(len(models))
axes_flat[2].plot(x_pos, vowel_and_consonant_correct, marker='o', linewidth=2, markersize=8, label='Downweighting EOL (at end of first line)')
axes_flat[2].plot(x_pos, neol_eventually_rhymed, marker='s', linewidth=2, markersize=8, label='Downweighting NEOL')
axes_flat[2].plot(x_pos, attn_eventually_rhymed, marker='s', linewidth=2, markersize=8, label='Ablating attention')
axes_flat[2].set_xticks(x_pos)
axes_flat[2].set_xticklabels(models)
axes_flat[2].set_title('Effects on rhyming accuracy')
axes_flat[2].text(0.92, 0.5, 'C', transform=axes_flat[2].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[2].set_ylabel('Rhyming Accuracy')
axes_flat[2].legend()
axes_flat[2].grid(True, alpha=0.3)

# Plot D: Rhyme Intervention Sample NC - Both Metrics
dfs_rhyme_sample = {model: pd.read_csv(f'rhyme_intervention_sample_nc/{model}.csv') for model in models}
vowel_and_consonant_correct = [(dfs_rhyme_sample[model]['vowel_correct'] & dfs_rhyme_sample[model]['consonant_correct']).mean() for model in models]
vowel_and_singular_consonant_correct = [(dfs_rhyme_sample[model]['vowel_correct'] & dfs_rhyme_sample[model]['singular_consonant_correct']).mean() for model in models]
x_pos = np.arange(len(models))
axes_flat[3].plot(x_pos, vowel_and_consonant_correct, marker='o', linewidth=2, markersize=8, label='Vowel & Consonant')
axes_flat[3].plot(x_pos, vowel_and_singular_consonant_correct, marker='s', linewidth=2, markersize=8, label='Vowel & Singular Consonant')
axes_flat[3].set_xticks(x_pos)
axes_flat[3].set_xticklabels(models)
axes_flat[3].set_title('Effectiveness of Changing Rhyming Features')
axes_flat[3].text(0.92, 0.5, 'D', transform=axes_flat[3].transAxes, fontsize=24, fontweight='bold', ha='center', va='center')
axes_flat[3].set_ylabel('Rhyming Accuracy (for new rhyme)')
axes_flat[3].legend()
axes_flat[3].grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.savefig('combined_circuit_plot_attn.pdf')
plt.show()
#%%