#!/usr/bin/env python3
"""
Probing script for TinyStories model representations.

This script:
1. Takes a model name as input
2. Loads the corresponding CSV file with results
3. Encodes detected_animal column for classification
4. Extracts model representations of last tokens from first_sentence
5. Trains a d-64 MLP probe to predict detected_animal
6. Reports classification results with 60/20/20 train/valid/test split
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MLPProbe(nn.Module):
    """Simple MLP probe with hidden dimension 64."""
    
    def __init__(self, input_dim, num_classes, hidden_dim=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    print(f"✓ Successfully loaded {model_name}")
    return model, tokenizer


def extract_last_token_representations(sentences, model, tokenizer, batch_size=32, layer_idx=-1):
    """Extract representations of the last token from each sentence."""
    representations = []
    model.eval()
    
    print(f"Extracting last token representations from layer {layer_idx}...")
    
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch_sentences = sentences[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  # Specified layer
            
            # Get last token representation for each sequence
            batch_representations = []
            for j, attention_mask in enumerate(inputs["attention_mask"]):
                # Find the last non-padding token
                last_token_idx = attention_mask.sum() - 1
                last_token_repr = hidden_states[j, last_token_idx, :]
                batch_representations.append(last_token_repr.cpu())
            
            representations.extend(batch_representations)
    
    return torch.stack(representations)


def train_probe(X_train, y_train, X_val, y_val, num_classes, epochs=400, lr=0.001):
    """Train MLP probe."""
    input_dim = X_train.shape[1]
    probe = MLPProbe(input_dim, num_classes)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = probe.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    
    best_val_acc = 0
    best_probe_state = None
    
    print("Training probe...")
    for epoch in tqdm(range(epochs)):
        probe.train()
        optimizer.zero_grad()
        
        outputs = probe(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            probe.eval()
            with torch.no_grad():
                val_outputs = probe(X_val)
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_acc = (val_predictions == y_val).float().mean().item()
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_probe_state = probe.state_dict().copy()
    
    # Load best model
    probe.load_state_dict(best_probe_state)
    return probe, best_val_acc


def evaluate_probe(probe, X_test, y_test, label_encoder):
    """Evaluate probe on test set."""
    device = next(probe.parameters()).device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    probe.eval()
    with torch.no_grad():
        outputs = probe(X_test)
        predictions = torch.argmax(outputs, dim=1)
        
    # Convert back to CPU for sklearn metrics
    predictions = predictions.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    accuracy = accuracy_score(y_test, predictions)
    
    # Get unique classes present in test data and their names
    unique_classes = np.unique(y_test)
    class_names = [label_encoder.classes_[i] for i in unique_classes]
    
    # Calculate F1 scores
    f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, predictions, average='micro', zero_division=0)
    f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    # Generate report only for classes present in test data
    report = classification_report(
        y_test, predictions, 
        labels=unique_classes,
        target_names=class_names, 
        zero_division=0
    )
    
    return accuracy, report, f1_macro, f1_micro, f1_weighted


def create_result_dict(model_name, layer, train_size, val_size, test_size, 
                      num_classes, representation_dim, best_val_acc, test_accuracy, 
                      f1_macro, f1_micro, f1_weighted, animal_classes):
    """Create result dictionary for JSON output."""
    return {
        "model": model_name,
        "layer": layer,
        "split_sizes": {
            "train": train_size,
            "validation": val_size,
            "test": test_size
        },
        "num_classes": num_classes,
        "animal_classes": animal_classes,
        "representation_dim": representation_dim,
        "validation_accuracy": best_val_acc,
        "test_accuracy": test_accuracy,
        "f1_scores": {
            "macro": f1_macro,
            "micro": f1_micro,
            "weighted": f1_weighted
        }
    }


def save_results_to_json(results, model_noslash, output_dir="probing"):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    json_file = output_path / f"{model_noslash}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {json_file}")
    return json_file


def run_single_layer_probe(sentences, y, label_encoder, model, tokenizer, args, layer_idx):
    """Run probing for a single layer."""
    # Extract representations
    X = extract_last_token_representations(sentences, model, tokenizer, args.batch_size, layer_idx)
    
    # Split data (60/20/20) with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
    )
    
    # Convert to tensors (ensure float32 for MLP training)
    X_train = X_train.float() 
    X_val = X_val.float()
    X_test = X_test.float() 
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    y_test = torch.LongTensor(y_test)
    
    # Train probe
    num_classes = len(label_encoder.classes_)
    probe, best_val_acc = train_probe(X_train, y_train, X_val, y_val, num_classes, args.epochs, args.lr)
    
    # Evaluate on test set
    test_accuracy, test_report, f1_macro, f1_micro, f1_weighted = evaluate_probe(probe, X_test, y_test, label_encoder)
    
    return {
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'representation_dim': X.shape[1],
        'best_val_acc': best_val_acc,
        'test_accuracy': test_accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'test_report': test_report
    }


def main():
    # Hardcoded configuration from run_tinystories_probing.sh
    EPOCHS = 400
    LR = 0.001
    BATCH_SIZE = 32
    ALL_LAYERS = True  # Equivalent to --all-layers flag
    
    # Hardcoded model list
    QWEN3_MODELS = [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B", 
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B"
    ]
    
    OTHER_MODELS = [
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]
    
    ALL_MODELS = QWEN3_MODELS + OTHER_MODELS
    TOTAL_MODELS = len(ALL_MODELS)
    
    print(f"Starting evaluation for {TOTAL_MODELS} models:")
    for i, model in enumerate(ALL_MODELS):
        print(f"  {i+1}. {model}")
    
    # Process all models
    for i, model_name in enumerate(ALL_MODELS):
        print("")
        print("=" * 60)
        print(f"[{i+1}/{TOTAL_MODELS}] Starting evaluation for: {model_name}")
        print("=" * 60)
        
        # Load data
        model_noslash = model_name.split('/')[-1]
        csv_path = Path(f"results/behavioral/{model_noslash}.csv")
        if not csv_path.exists():
            print(f"Error: CSV file not found at {csv_path}")
            continue
        
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Remove rows with missing animals
        df = df.dropna(subset=['detected_animal'])
        print(f"Dataset size: {len(df)} examples")
        
        # Check class distribution and keep only top 4 most common animals
        animal_counts = df['detected_animal'].value_counts()
        print(f"Animal distribution:\n{animal_counts}")
        
        # Keep only the 4 most common animals
        top_4_animals = animal_counts.head(4).index
        df_filtered = df[df['detected_animal'].isin(top_4_animals)].copy()
        
        print(f"Top 4 animals: {list(top_4_animals)}")
        print(f"Filtered dataset: {len(df_filtered)} examples (removed {len(df) - len(df_filtered)} examples)")
        print(f"Removed animals: {set(animal_counts.index) - set(top_4_animals)}")
        
        # Encode animals
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df_filtered['detected_animal'])
        num_classes = len(label_encoder.classes_)
        print(f"Number of animal classes after filtering: {num_classes}")
        print(f"Animal classes: {list(label_encoder.classes_)}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name)
        
        sentences = df_filtered['first_sentence'].tolist()
        
        # Create a simple args object for compatibility
        class Args:
            def __init__(self):
                self.epochs = EPOCHS
                self.lr = LR
                self.batch_size = BATCH_SIZE
        
        args = Args()
        
        if ALL_LAYERS:
            # Get the number of layers in the model
            num_layers = len(model.model.layers) if hasattr(model.model, 'layers') else len(model.transformer.h)
            print(f"Model has {num_layers} layers. Running probing on all layers...")
            
            all_results = []
            
            for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
                print(f"\n{'='*20} Layer {layer_idx} {'='*20}")
                
                result = run_single_layer_probe(sentences, y, label_encoder, model, tokenizer, args, layer_idx)
                
                # Create result dictionary for this layer
                layer_result = create_result_dict(
                    model_name, layer_idx, result['train_size'], result['val_size'], result['test_size'],
                    num_classes, result['representation_dim'], result['best_val_acc'], result['test_accuracy'],
                    result['f1_macro'], result['f1_micro'], result['f1_weighted'], list(label_encoder.classes_)
                )
                
                all_results.append(layer_result)
                
                print(f"Layer {layer_idx} - Test accuracy: {result['test_accuracy']:.4f}, F1 (macro): {result['f1_macro']:.4f}")
            
            # Save all results to JSON
            save_results_to_json(all_results, model_noslash)
            
            print(f"\n{'='*50}")
            print("ALL LAYERS PROBING COMPLETED")
            print(f"{'='*50}")
            print(f"Results for all {num_layers} layers saved to JSON")
        
        # Clean up model to free memory before next iteration
        del model
        del tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()