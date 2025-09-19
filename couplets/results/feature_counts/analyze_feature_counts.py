#%%
import torch
import matplotlib.pyplot as plt
# %%
# Get the data for plotting
def make_plot_from_feature_data(feature_data, last_word: str, count_name='selected_counts', combine_word_features=False, only_second_line=False, model=''):
    input_tokens_str = feature_data['input_tokens_str']
    counts = feature_data[count_name]
    if only_second_line:
        end_think_idx = input_tokens_str.index('</think>')
        input_tokens_str  = input_tokens_str[end_think_idx + 2:]
        counts = {k: v[end_think_idx + 2:] for k,v in counts.items()}

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

    plt.title(f'Feature Counts by Input Tokens - {model} - {last_word}')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(input_tokens_str, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
#%%
def make_rhyme_plot_from_feature_data(feature_data, last_word: str, count_name='selected_counts', model = ''):
    input_tokens_str = feature_data['input_tokens_str']
    counts = feature_data[count_name]

    # Extract all count types
    near_eol_count = counts['near_eol_feature_count'].numpy()
    rhyme_feature_count = counts['last_rhyme_feature_count'].numpy()

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
    lines = line1 + line2

    # Combine legends from both axes
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Feature Counts by Input Tokens - {model} - {last_word}')
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
    make_plot_from_feature_data(feature_data, name.split('-'), 'selected_counts', combine_word_features=True, only_second_line=True, model=model)

# %%
for name, feature_data in data.items():
    make_rhyme_plot_from_feature_data(feature_data, name.split('-'), 'selected_counts', model=model)
# %%
def get_word_count(feature_data, count_type):
    counts = feature_data[count_type]
    say_x_count = counts['say_x_feature_count'].numpy()[-2]
    word_count = counts['word_feature_count'].numpy()[-2]
    near_word_count = counts['near_word_feature_count'].numpy()[-2]
    return say_x_count + word_count + near_word_count

#%%
from collections import defaultdict

c = defaultdict(list)
for name, feature_data in data.items():
    _, last_word = name.split('-')
    c[last_word].append(get_word_count(feature_data, 'selected_counts'))

#%%
sorted_counts = sorted(c.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
# %%
print(sorted_counts)
# %%
[name for name in data.keys() if name.split('-')[1] == 'night']
# %%
