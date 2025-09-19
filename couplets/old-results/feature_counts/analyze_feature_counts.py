#%%
import torch
import matplotlib.pyplot as plt
# %%
# Get the data for plotting
def make_plot_from_feature_data(feature_data, count_name='selected_counts', combine_word_features=False):
    input_tokens_str = feature_data['input_tokens_str']
    counts = feature_data[count_name]

    # Extract all count types
    near_eol_count = counts['near_eol_feature_count'].numpy()
    say_x_count = counts['say_x_feature_count'].numpy()
    word_count = counts['word_feature_count'].numpy()
    near_word_count = counts['near_word_feature_count'].numpy()

    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 4))

    x_positions = range(len(input_tokens_str))

    # Plot near_eol_count on the left axis (primary)
    color1 = 'tab:blue'
    ax1.set_xlabel('Input Tokens')
    ax1.set_ylabel('Near EOL Feature Count', color=color1)
    line1 = ax1.plot(x_positions, near_eol_count, marker='o', color=color1, label='Near EOL Feature Count', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for other counts
    ax2 = ax1.twinx()
    ax2.set_ylabel('Other Feature Counts', color='black')
    ax2.tick_params(axis='y')

    if combine_word_features:
        # Combine all other features into one line
        combined_count = say_x_count + word_count + near_word_count
        line2 = ax2.plot(x_positions, combined_count, marker='s', color='tab:red', label='Combined Word Features', linewidth=2)
        lines = line1 + line2
    else:
        # Plot each feature separately
        color2 = 'tab:red'
        color3 = 'tab:green'
        color4 = 'tab:orange'
        
        line2 = ax2.plot(x_positions, say_x_count, marker='s', color=color2, label='Say X Feature Count', linewidth=2)
        line3 = ax2.plot(x_positions, word_count, marker='^', color=color3, label='Word Feature Count', linewidth=2)
        line4 = ax2.plot(x_positions, near_word_count, marker='d', color=color4, label='Near Word Feature Count', linewidth=2)
        lines = line1 + line2 + line3 + line4

    # Combine legends from both axes
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Feature Counts by Input Tokens - {model}')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(input_tokens_str, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# %%
models = [f'Qwen3-{size}B' for size in [0.6,1.7,4,8,14]]

model = models[-1]

data = torch.load(f'{model}.pt')

#%%
for name, feature_data in data.items():
    make_plot_from_feature_data(feature_data, 'selected_counts',combine_word_features=True)
#%%
def make_rhyme_plot_from_feature_data(feature_data, count_name='selected_counts'):
    input_tokens_str = feature_data['input_tokens_str']
    counts = feature_data[count_name]

    # Extract all count types
    near_eol_count = counts['near_eol_feature_count'].numpy()
    rhyme_feature_count = counts['rhyme_feature_count'].numpy()

    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 4))

    x_positions = range(len(input_tokens_str))

    # Plot near_eol_count on the left axis (primary)
    color1 = 'tab:blue'
    ax1.set_xlabel('Input Tokens')
    ax1.set_ylabel('Near EOL Feature Count', color=color1)
    line1 = ax1.plot(x_positions, near_eol_count, marker='o', color=color1, label='Near EOL Feature Count', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for other counts
    ax2 = ax1.twinx()
    ax2.set_ylabel('Other Feature Counts', color='black')
    ax2.tick_params(axis='y')


    # Combine all other features into one line
    line2 = ax2.plot(x_positions, rhyme_feature_count, marker='s', color='tab:red', label='Rhyme Features', linewidth=2)
    lines = line2

    # Combine legends from both axes
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Feature Counts by Input Tokens - {model}')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(input_tokens_str, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
# %%
for name, feature_data in data.items():
    make_rhyme_plot_from_feature_data(feature_data, 'pruned_counts')
# %%
