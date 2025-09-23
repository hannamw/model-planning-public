#%%
import torch
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

models_to_transcoders = {
    'Qwen3-0.6B':"mwhanna/qwen3-0.6b-transcoders-lowl0",
    'Qwen3-1.7B':"mwhanna/qwen3-1.7b-transcoders-lowl0",
    'Qwen3-4B':"mwhanna/qwen3-4b-transcoders",
    'Qwen3-8B':"mwhanna/qwen3-8b-transcoders",
    'Qwen3-14B':"mwhanna/qwen3-14b-transcoders-lowl0"
}

def load_saved_results(models):
    """Load saved results from disk"""
    loaded_results = {}
    
    for model in models:
        model_name_parts = model.split('-')
        size_part = model_name_parts[1].upper()
        if size_part.endswith('B'):
            size_part = size_part[:-1] + 'B'
        logit_lens_model_name = f"Qwen3-{size_part}"
        
        results_file = f'{logit_lens_model_name}.pt'
        loaded_results[model] = torch.load(results_file, weights_only=False)
        print(f"Loaded results for {model} from {results_file}")
    
    return loaded_results

models = list(models_to_transcoders.keys())
all_model_results = load_saved_results(models)

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
        model_names.append(model)
        results = all_model_results[model]
        metadata = results['metadata']
        per_example_influences = results['per_example_cumsum_selected_influences'][:, -1]

        # Extract influences for each category using the correct method
        a_corrects = per_example_influences[(metadata['article_correct']) & (metadata['article'] == 'a')]
        a_incorrects = per_example_influences[(~metadata['article_correct']) & (metadata['article'] == 'a')]
        an_corrects = per_example_influences[(metadata['article_correct']) & (metadata['article'] == 'an')]
        an_incorrects = per_example_influences[(~metadata['article_correct']) & (metadata['article'] == 'an')]
        
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
    
    # Create bars with different colors for a/an and dots for correct/incorrect
    # Blue for 'a', purple for 'an'
    # Solid bars for correct, dotted pattern for incorrect
    a_color = '#2E86AB'  # Blue for "a"
    an_color = '#A23B72'  # Purple for "an"
    
    bars1 = ax.bar(x_pos - 1.5*bar_width, a_correct_means, bar_width, 
                   color=a_color, alpha=0.8, label='"a" Correct', 
                   edgecolor='black', linewidth=0.5)
    
    bars2 = ax.bar(x_pos - 0.5*bar_width, a_incorrect_means, bar_width,
                   color=a_color, alpha=0.8, label='"a" Incorrect',
                   hatch='///', edgecolor='black', linewidth=0.5)
    
    bars3 = ax.bar(x_pos + 0.5*bar_width, an_correct_means, bar_width,
                   color=an_color, alpha=0.8, label='"an" Correct',
                   edgecolor='black', linewidth=0.5)
    
    bars4 = ax.bar(x_pos + 1.5*bar_width, an_incorrect_means, bar_width,
                   color=an_color, alpha=0.8, label='"an" Incorrect', 
                   hatch='///', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_ylabel('Prop. of Influence via Planning Nodes', fontsize=18)
    ax.set_title('Mean Proportion of Influence Through Planning Nodes', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=20)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='upper left', fontsize=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('a_an_influence.pdf')
    plt.show()

# Create the combined bar plot
create_combined_mean_influence_bar_plot(all_model_results, models)

# %%
