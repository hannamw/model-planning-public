#%%
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from circuit_tracer.graph import Graph
from load_important_nodes import convert_model_name

# List of models to analyze
models = ['qwen3-0.6b-relu-lowl0', 'qwen3-1.7b-relu-lowl0', 'qwen3-4b-relu', 'qwen3-8b-relu', 'qwen3-14b-relu-lowl0']

# Define threshold parameters
REQUIRED_PROFESSION_COUNT = 5
REQUIRED_RELATED_TERMS_COUNT = 10

def load_important_nodes(model_name: str, example_key: str) -> list:
    """Load and filter important nodes based on profession and related terms counts
    
    Args:
        model_name: Name of the model (e.g. 'qwen3-0.6b-relu-lowl0')
        example_key: Example identifier (e.g. 'a-archaeologist')
    
    Returns:
        List of [layer, feature, pos] lists for important nodes, or None if no nodes found
    """
    # Convert model name format
    model_name_parts = model_name.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    qwen_name = f"Qwen3-{size_part}"
    
    # Load relevant nodes
    relevant_nodes_path = f'results/relevant_nodes_refined/{qwen_name}/{example_key}.json'
    with open(relevant_nodes_path) as f:
        relevant_nodes = json.load(f)

    # Filter nodes by profession and related terms counts
    filtered_nodes = []
    for node_id, data in relevant_nodes['feature_counts'].items():
        if (data['profession_count'] > REQUIRED_PROFESSION_COUNT or 
            data['related_terms_count'] > REQUIRED_RELATED_TERMS_COUNT):
            # Convert node_id format from layer_pos_feature to [layer, pos, feature]
            layer, pos, feature = map(int, node_id.split('_'))
            filtered_nodes.append([layer, pos, feature])
    
    return filtered_nodes if filtered_nodes else None

def analyze_model(model: str):
    """Analyze relationship between important nodes and performance for a model"""
    # Load all examples for this model
    data = []
    
    # Convert model name for metadata lookup
    model_name_parts = model.split('-')
    size_part = model_name_parts[1].upper()  # 0.6b -> 0.6B
    if size_part.endswith('B'):
        size_part = size_part[:-1] + 'B'
    logit_lens_model = f"Qwen3-{size_part}"
    
    # Load metadata
    metadata = pd.read_csv(f'results/logit-lens/{logit_lens_model}/metadata.csv')
    
    # Process each example
    for _, row in metadata.iterrows():
        article = row['correct_articles']
        profession = row['professions']
        key = f"{article}-{profession}"
        
        # Load graph and important nodes
        graph_file = Path('attribution_graphs') / model / f"{key}.pt"
            
        graph = Graph.from_pt(str(graph_file))
        selected_nodes = load_important_nodes(model, key)

        # Create selected nodes mask if provided
        if selected_nodes:
            candidate_features = graph.active_features[graph.selected_features].unsqueeze(0)
            selected_features = torch.tensor(selected_nodes)
            selected_features_us = selected_features.unsqueeze(1)
            matches = torch.all(candidate_features == selected_features_us, dim=2)
            selected_nodes_mask = torch.any(matches, dim=1)  
            selected_nodes = selected_features[selected_nodes_mask].tolist()
             
        if selected_nodes:
            # Filter nodes to only those that exist in the graph at the last position
            filtered_nodes = []
            for node in selected_nodes:
                layer, pos, feature = node
                if pos == graph.n_pos - 1:  # Only keep nodes at the last position
                    filtered_nodes.append(node)
            
            if filtered_nodes:
                # Get probability of correct answer
                p_correct = row['p(a)'] if article == 'a' else row['p(an)']
                
                data.append({
                    'model': model,
                    'article': article,
                    'profession': profession,
                    'num_nodes': len(filtered_nodes),
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
# %%
