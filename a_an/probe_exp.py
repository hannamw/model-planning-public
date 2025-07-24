#%%
from transformer_lens import HookedTransformer
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import namedtuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import create_dataset_examples

Example = namedtuple("Example", ["sentence", "continuation", "name"])

def get_residual_representations(model, cache):
    """Get residual representations from all layers"""
    n_layers = model.cfg.n_layers
    representations = []
    
    for layer in range(n_layers):
        # Get residual stream at this layer
        residual = cache["resid_post", layer]
        # Get the last token position
        last_token_repr = residual.squeeze()[-1]  # Shape: [d_model]
        representations.append(last_token_repr.cpu())
    
    return torch.stack(representations)  # Shape: [n_layers, d_model]

def train_probe_on_layer(X_train, y_train, X_test, y_test):
    """Train a linear probe and return accuracy"""
    probe = LogisticRegression(random_state=42, max_iter=1000)
    probe.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))
    
    return train_acc, test_acc, probe



# Model configurations (based on compute_a_an_graphs.py)
model_names = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B',
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-14B'
]

# Load dataset
df = pd.read_csv('data/professions_dataset_with_articles.csv')
df_ex = create_dataset_examples(df)

# Create output directory
output_dir = Path('results/probe_analysis')
output_dir.mkdir(exist_ok=True)

for model_name in model_names:
    print(f"Processing {model_name}...")
    
    # Load model using HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16
    )
    
    # Create examples as in compute_a_an_graphs.py
    examples = [Example(f"{model.tokenizer.eos_token}{prompt}", f' {article}', f'{article}-{profession}') 
                for prompt, article, profession in zip(df_ex['Prompt'], df_ex['Article'], df_ex['Profession'])]
    
    n_examples = len(examples)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # Initialize storage
    all_representations = torch.zeros((n_examples, n_layers, d_model))
    labels = []
    correct_articles = []
    professions = []
    sentences = []
    model_predictions = []
    
    # Process each example
    for i, (sentence, continuation, name) in enumerate(examples):
        print(f"  Processing example {i+1}/{n_examples}: {name}")
        
        with torch.inference_mode():
            # Run model with cache to get residual stream at all layers
            logits, cache = model.run_with_cache(sentence)
            
            # Get residual representations for all layers
            representations = get_residual_representations(model, cache)
            all_representations[i] = representations
            
            # Get model predictions from existing logits
            probs = F.softmax(logits.squeeze()[-1], dim=-1)
            a_token_id = model.tokenizer.encode(" a", add_special_tokens=False)[0]
            an_token_id = model.tokenizer.encode(" an", add_special_tokens=False)[0]
            a_prob = probs[a_token_id].item()
            an_prob = probs[an_token_id].item()
            
            # Top-1 prediction correctness
            top1_token_id = torch.argmax(probs).item()
            if top1_token_id == a_token_id:
                model_pred = "a"
            elif top1_token_id == an_token_id:
                model_pred = "an"
            else:
                model_pred = "other"
            model_predictions.append(model_pred)
            
            # Store metadata
            correct_article = name.split('-')[0]
            profession = name.split('-')[1]
            
            labels.append(0 if correct_article == "a" else 1)  # 0 for "a", 1 for "an"
            correct_articles.append(correct_article)
            professions.append(profession)
            sentences.append(sentence)
    
    # Convert to numpy for sklearn
    labels = np.array(labels)
    
    # Train probes for each layer
    layer_accuracies = []
    train_accuracies = []
    
    # Split data into train/test
    X_train_indices, X_test_indices = train_test_split(
        range(n_examples), test_size=0.2, random_state=42, stratify=labels
    )
    
    for layer in range(n_layers):
        print(f"  Training probe for layer {layer+1}/{n_layers}")
        
        # Get representations for this layer
        layer_repr = all_representations[:, layer, :].numpy()  # Shape: [n_examples, d_model]
        
        # Split into train/test
        X_train = layer_repr[X_train_indices]
        X_test = layer_repr[X_test_indices]
        y_train = labels[X_train_indices]
        y_test = labels[X_test_indices]
        
        # Train probe
        train_acc, test_acc, probe = train_probe_on_layer(X_train, y_train, X_test, y_test)
        
        train_accuracies.append(train_acc)
        layer_accuracies.append(test_acc)
    
    # Create model-specific output directory
    model_short_name = model_name.split('/')[-1]
    model_output_dir = output_dir / model_short_name
    model_output_dir.mkdir(exist_ok=True)
    
    # Create metadata with model predictions
    metadata_df = pd.DataFrame({
        'sentences': sentences,
        'correct_articles': correct_articles,
        'professions': professions,
        'model_predictions': model_predictions,
        'model_correct': [pred == correct for pred, correct in zip(model_predictions, correct_articles)],
        'labels': labels
    })
    
    # Save results
    results_dict = {
        'layer_accuracies': layer_accuracies,
        'train_accuracies': train_accuracies,
        'representations': all_representations,
        'labels': labels,
        'X_train_indices': X_train_indices,
        'X_test_indices': X_test_indices
    }
    
    torch.save(results_dict, model_output_dir / 'probe_results.pt')
    metadata_df.to_csv(model_output_dir / 'metadata.csv', index=False)
    
    print(f"  Saved results for {model_name} to {model_output_dir}")
    print(f"  Best probe accuracy: {max(layer_accuracies):.3f} at layer {np.argmax(layer_accuracies) + 1}")
    
    # Free memory
    del model
    torch.cuda.empty_cache()

print(f"\nAll results saved to {output_dir}") 

# %%
# Load and create detailed plots for each model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Load all results
output_dir = Path('results/probe_analysis')
model_names = [
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-1.7B', 
    'Qwen/Qwen3-4B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-14B'
]

all_results = {}
all_metadata = {}

for model_name in model_names:
    model_short_name = model_name.split('/')[-1]
    model_dir = output_dir / model_short_name
    
    if model_dir.exists():
        # Load results
        results = torch.load(model_dir / 'probe_results.pt', weights_only=False)
        metadata = pd.read_csv(model_dir / 'metadata.csv')
        
        all_results[model_short_name] = results
        all_metadata[model_short_name] = metadata
        
        print(f"Loaded {model_short_name}: {len(results['layer_accuracies'])} layers")

# Create individual detailed plots for each model
for model_name in all_results:
    metadata = all_metadata[model_name]
    results = all_results[model_name]
    n_layers = len(results['layer_accuracies'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Detailed Analysis: {model_name}', fontsize=14, fontweight='bold')
    
    # Main accuracy plot
    ax1 = axes[0]
    ax1.plot(range(1, n_layers + 1), results['layer_accuracies'], 'b-', label='Test Accuracy', linewidth=2)
    ax1.plot(range(1, n_layers + 1), results['train_accuracies'], 'r--', label='Train Accuracy', linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Probe Accuracy vs Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best layer info
    ax2 = axes[1]
    best_layer = np.argmax(results['layer_accuracies']) + 1
    best_acc = max(results['layer_accuracies'])
    model_acc = metadata['model_correct'].mean()
    
    ax2.text(0.1, 0.8, f'Best Layer: {best_layer}', fontsize=14, fontweight='bold')
    ax2.text(0.1, 0.7, f'Best Accuracy: {best_acc:.3f}', fontsize=14, fontweight='bold')
    ax2.text(0.1, 0.6, f'Total Examples: {len(metadata)}', fontsize=12)
    ax2.text(0.1, 0.5, f'Train Examples: {len(results["X_train_indices"])}', fontsize=12)
    ax2.text(0.1, 0.4, f'Test Examples: {len(results["X_test_indices"])}', fontsize=12)
    ax2.text(0.1, 0.3, f'Model Accuracy: {model_acc:.3f}', fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_detailed_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

print(f"\nIndividual model plots saved for each model to {output_dir}")

# %%
