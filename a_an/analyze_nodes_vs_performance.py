#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_important_nodes import load_important_nodes_for_model, convert_model_name

# List of models to analyze
models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']

def analyze_model(model: str):
    """Analyze relationship between important nodes and performance for a model"""
    # Load all examples for this model
    results = load_important_nodes_for_model(
        model=model,
        count_threshold=5,
        similarity_threshold=0.63,
        position_filter=-1
    )
    
    # Extract data for each example
    data = []
    for example_key, result in results.items():
        article, profession = example_key.split('-')
        num_nodes = result['num_relevant_features']
        
        # Get metadata about correctness and probabilities
        logit_lens_model = convert_model_name(model)
        metadata = pd.read_csv(f'results/logit-lens/{logit_lens_model}/metadata.csv')
        row = metadata[
            (metadata['correct_articles'] == article) & 
            (metadata['professions'] == profession)
        ].iloc[0]
        
        # Get probability of correct answer
        p_correct = row['p(a)'] if article == 'a' else row['p(an)']
        
        data.append({
            'model': model,
            'article': article,
            'profession': profession,
            'num_nodes': num_nodes,
            'p_correct': p_correct,
            'is_correct': row['correct?']
        })
    
    return pd.DataFrame(data)

#%%
# Collect data for all models
all_data = pd.concat([analyze_model(model) for model in models], ignore_index=True)

#%%
# Plot 1: Number of nodes vs p(correct) for all examples
plt.figure(figsize=(12, 8))
for model in models:
    model_data = all_data[all_data['model'] == model]
    plt.scatter(model_data['num_nodes'], model_data['p_correct'], 
                label=convert_model_name(model), alpha=0.6)

plt.xlabel('Number of Important Nodes')
plt.ylabel('P(Correct Answer)')
plt.title('Number of Important Nodes vs. Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
# Plot 2: Separate plots for 'a' and 'an' examples
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

for model in models:
    model_data = all_data[all_data['model'] == model]
    
    # Plot for 'a' examples
    a_data = model_data[model_data['article'] == 'a']
    ax1.scatter(a_data['num_nodes'], a_data['p_correct'], 
                label=convert_model_name(model), alpha=0.6)
    
    # Plot for 'an' examples
    an_data = model_data[model_data['article'] == 'an']
    ax2.scatter(an_data['num_nodes'], an_data['p_correct'], 
                label=convert_model_name(model), alpha=0.6)

ax1.set_xlabel('Number of Important Nodes')
ax1.set_ylabel('P(Correct Answer)')
ax1.set_title('Performance vs Nodes for "a" Examples')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Number of Important Nodes')
ax2.set_ylabel('P(Correct Answer)')
ax2.set_title('Performance vs Nodes for "an" Examples')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Plot 3: Add trend lines using seaborn
plt.figure(figsize=(12, 8))
for model in models:
    model_data = all_data[all_data['model'] == model]
    sns.regplot(data=model_data, x='num_nodes', y='p_correct', 
                label=convert_model_name(model), scatter=True,
                scatter_kws={'alpha':0.3}, line_kws={'alpha':0.7})

plt.xlabel('Number of Important Nodes')
plt.ylabel('P(Correct Answer)')
plt.title('Number of Important Nodes vs. Performance (with Trend Lines)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#%%
# Plot 4: Separate trend lines for 'a' and 'an' examples
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

for model in models:
    model_data = all_data[all_data['model'] == model]
    
    # Plot for 'a' examples with trend line
    a_data = model_data[model_data['article'] == 'a']
    sns.regplot(data=a_data, x='num_nodes', y='p_correct',
                label=convert_model_name(model), scatter=True,
                scatter_kws={'alpha':0.3}, line_kws={'alpha':0.7}, ax=ax1)
    
    # Plot for 'an' examples with trend line
    an_data = model_data[model_data['article'] == 'an']
    sns.regplot(data=an_data, x='num_nodes', y='p_correct',
                label=convert_model_name(model), scatter=True,
                scatter_kws={'alpha':0.3}, line_kws={'alpha':0.7}, ax=ax2)

ax1.set_xlabel('Number of Important Nodes')
ax1.set_ylabel('P(Correct Answer)')
ax1.set_title('Performance vs Nodes for "a" Examples (with Trend Lines)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Number of Important Nodes')
ax2.set_ylabel('P(Correct Answer)')
ax2.set_title('Performance vs Nodes for "an" Examples (with Trend Lines)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Print summary statistics
print("\nSummary Statistics:")
print("==================")

for model in models:
    model_data = all_data[all_data['model'] == model]
    print(f"\n{convert_model_name(model)}:")
    print(f"Average nodes: {model_data['num_nodes'].mean():.2f}")
    print(f"Average p(correct): {model_data['p_correct'].mean():.3f}")
    
    # Split by article
    a_data = model_data[model_data['article'] == 'a']
    an_data = model_data[model_data['article'] == 'an']
    
    print(f"'a' examples - avg nodes: {a_data['num_nodes'].mean():.2f}, avg p(correct): {a_data['p_correct'].mean():.3f}")
    print(f"'an' examples - avg nodes: {an_data['num_nodes'].mean():.2f}, avg p(correct): {an_data['p_correct'].mean():.3f}") 