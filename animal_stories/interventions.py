#!/usr/bin/env python3
"""
Intervention script for TinyStories model representations.

This script:
1. Loads dataset and filters to top 4 animals (same as probing file)
2. Creates pairs of first sentences that produce different animals
3. Performs activation patching interventions
4. Records generated continuations and detected animals
"""

import pandas as pd
import re
import torch
from pathlib import Path
from transformer_lens import HookedTransformer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_animals_list(animals_file: Path = Path("data/animals.txt")) -> set[str]:
    """Load the animals list from file and return as a set of lowercase names."""
    animals = set()
    
    with open(animals_file, 'r', encoding='utf-8') as f:
        for line in f:
            animal = line.strip()
            if animal:  # Skip empty lines
                animals.add(animal.lower())
    
    print(f"Loaded {len(animals)} animals from {animals_file}")
    return animals


def detect_animal_in_text(text: str, animals_set: set[str]) -> str:
    """
    Detect which animal is mentioned in the text.
    Returns the animal name (original case) or empty string if none found.
    """
    if not text or not animals_set:
        return ""
    
    # Convert text to lowercase for matching
    text_lower = text.lower()
    
    # Sort animals by length (longest first) to prioritize longer matches
    sorted_animals = sorted(animals_set, key=len, reverse=True)
    
    for animal in sorted_animals:
        # Use word boundaries to avoid partial matches
        if ' ' in animal:
            # Multi-word animal - look for the phrase
            if animal in text_lower:
                pattern = r'\b' + re.escape(animal) + r'\b'
                if re.search(pattern, text_lower):
                    return animal.title()
        else:
            # Single word animal - use strict word boundaries
            pattern = r'\b' + re.escape(animal) + r'\b'
            if re.search(pattern, text_lower):
                return animal.title()
    
    return ""


def clean_response(response: str, first_sentence: str = "") -> str:
    """Clean the generated response by removing extra formatting."""
    # Remove complete <think>...</think> tags and their content
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
    response = re.sub(r'</think>', '', response)
    
    # Strip whitespace
    response = response.strip()
    
    # Remove the first sentence if it's repeated in the continuation
    if first_sentence and response:
        first_clean = first_sentence.rstrip('.!?').strip()
        if response.lower().startswith(first_clean.lower()):
            response = response[len(first_clean):].strip()
            response = response.lstrip('.!?').strip()
    
    return response


def load_and_filter_data(model_name: str) -> pd.DataFrame:
    """Load data and filter to top 4 animals (same approach as probing file)."""
    model_noslash = model_name.split('/')[-1]
    csv_path = Path(f"results/tinystories/{model_noslash}.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
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
    print(f"Filtered dataset: {len(df_filtered)} examples")
    print(f"Removed animals: {set(animal_counts.index) - set(top_4_animals)}")
    
    return df_filtered


def create_sentence_pairs(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """
    Create pairs of first sentences that produce different animals.
    For each sentence, sample one sentence from all sentences leading to other animals.
    Returns list of tuples: (sentence1, animal1, sentence2, animal2)
    """    
    print(f"Creating sentence pairs from {len(df['detected_animal'].unique())} different animals")
    
    pairs = []
    
    # For each row, sample a random row with a different animal
    for idx, row in df.iterrows():
        # Get all rows with different animals
        other_animals_df = df[df['detected_animal'] != row['detected_animal']]
        
        # Sample one random row from other animals
        sampled_row = other_animals_df.sample(n=1).iloc[0]
        pairs.append((
            row['first_sentence'], 
            row['detected_animal'],
            sampled_row['first_sentence'], 
            sampled_row['detected_animal']
        ))
    
    print(f"Created {len(pairs)} sentence pairs")
    return pairs


def main():
    """Main function to run the intervention experiment."""
    # Hardcoded model list from run_tinystories_intervention.sh
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
    
    # Run interventions for all models
    for i, model_name in enumerate(ALL_MODELS):
        print("")
        print("=" * 60)
        print(f"[{i+1}/{TOTAL_MODELS}] Starting evaluation for: {model_name}")
        print("=" * 60)
        
        # Load data and model
        print("Loading dataset and model...")
        df_filtered = load_and_filter_data(model_name)
        model = HookedTransformer.from_pretrained(model_name)
        animals_set = load_animals_list()

        # Create sentence pairs
        sentence_pairs = create_sentence_pairs(df_filtered)

        # Prompt template
        prompt = """/no_think Here's the first sentence of a story: {first_sentence} Continue this story with one sentence that introduces a new animal character."""

        print(f"Starting interventions on {len(sentence_pairs)} pairs...")
        
        # Run the intervention experiment
        results = run_intervention_experiment(sentence_pairs, model, prompt, animals_set)

        # Save results
        results_df = pd.DataFrame(results)
        output_file = Path(f"results/tinystories_interventions/{model_name.split('/')[-1]}.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_file, index=False)

        # Print summary statistics
        print("\n" + "="*60)
        print("INTERVENTION EXPERIMENT RESULTS")
        print("="*60)
        print(f"Total pairs processed: {len(results)}")
        print(f"Successful interventions: {sum(r['intervention_successful'] for r in results)}")
        print(f"Success rate: {sum(r['intervention_successful'] for r in results) / len(results):.2%}")

        # Show breakdown by target animal
        target_animals = results_df['target_animal'].value_counts()
        print(f"\nTarget animal distribution:")
        for animal, count in target_animals.items():
            success_count = len(results_df[(results_df['target_animal'] == animal) & (results_df['intervention_successful'])])
            success_rate = success_count / count if count > 0 else 0
            print(f"  {animal}: {success_count}/{count} ({success_rate:.1%})")

        # Show some examples
        print(f"\nExample interventions:")
        for j, row in results_df.head(5).iterrows():
            print(f"\nPair {row['pair_id']}:")
            print(f"  Source: '{row['source_sentence']}' (expected: {row['source_animal']})")
            print(f"  Target: '{row['target_sentence']}' (expected: {row['target_animal']})")
            print(f"  Generated: '{row['generated_continuation']}'")
            print(f"  Detected: {row['detected_animal']}")
            print(f"  Success: {'✓' if row['intervention_successful'] else '✗'}")

        print(f"\nResults saved to: {output_file}")
        print("Experiment completed!")
        
        # Clean up model to free memory before next iteration
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_intervention_experiment(
    sentence_pairs: list[tuple[str, str, str, str]], 
    model: HookedTransformer, 
    prompt: str,
    animals_set: set[str]
) -> list[dict]:
    """
    Run intervention experiment on sentence pairs.
    For each pair, patch activations from second sentence to first sentence.
    """
    results = []
    
    for i, (sentence1, animal1, sentence2, animal2) in enumerate(tqdm(sentence_pairs, desc="Running interventions")):
        try:
            # Create messages for both sentences
            messages1 = [
                {'role': 'user', 'content': prompt.format(first_sentence=sentence1)}
            ]
            messages2 = [
                {'role': 'user', 'content': prompt.format(first_sentence=sentence2)}
            ]
            
            # Convert to input strings with generation prompt
            inputs1 = model.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
            inputs2 = model.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
            
            # Get activations from second sentence (source)
            logits2, cache2 = model.run_with_cache(inputs2)
            
            # Define hook function to patch activations from sentence2 to sentence1
            def replacement_hook(activations, hook):
                # Patch the last token position with activations from sentence2
                activations[:, -1] = cache2[hook.name][:, -1]
                return activations
            
            def remove_all_hook(activations, hook):
                model.remove_all_hook_fns()
            
            # Set up hooks for all layers
            hooks = []
            for layer in range(model.cfg.n_layers):
                hooks.append((f'blocks.{layer}.hook_resid_pre', replacement_hook))
                hooks.append((f'blocks.{layer}.hook_resid_post', replacement_hook))
            
            # Add hook to remove all hooks after generation
            hooks.append(('ln_final.hook_normalized', remove_all_hook))
            
            # Generate with intervention
            with model.hooks(hooks):
                generation = model.generate(
                    inputs1, 
                    max_new_tokens=30, 
                    do_sample=False,
                    temperature=0.0
                )
            
            # Clean the generated text
            generated_text = generation.replace(inputs1, "").strip()
            cleaned_generation = clean_response(generated_text, sentence1)
            
            # Detect animal in generated continuation
            detected_animal = detect_animal_in_text(cleaned_generation, animals_set)
            
            # Record result
            result = {
                'pair_id': i,
                'source_sentence': sentence1,
                'source_animal': animal1,
                'target_sentence': sentence2,
                'target_animal': animal2,
                'original_generation': generation,
                'generated_continuation': cleaned_generation,
                'detected_animal': detected_animal,
                'intervention_successful': detected_animal.lower() == animal2.lower() if detected_animal else False
            }
            
            results.append(result)
            
            # Print progress every 10 interventions
            if (i + 1) % 10 == 0:
                success_rate = sum(r['intervention_successful'] for r in results) / len(results)
                print(f"Processed {i+1}/{len(sentence_pairs)} pairs. Success rate: {success_rate:.2%}")
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            continue
    
    return results


if __name__ == "__main__":
    main()
