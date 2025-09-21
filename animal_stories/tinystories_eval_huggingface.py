#!/usr/bin/env python3
"""
Evaluate a single model on TinyStories dataset using Hugging Face transformers.

This script:
1. Loads the TinyStories dataset from Hugging Face
2. Extracts the first sentence from each story
3. Prompts the model to generate one more sentence (continuation)
4. Uses Hugging Face transformers for inference
5. Allows sampling a subset of examples
6. Saves results for analysis
7. Detects which animal is mentioned in the continuation
"""

import json
import gc
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 50    # Enough for one sentence
RESULTS_DIR = Path("results/tinystories")
ANIMALS_FILE = Path("../data/animals.txt")


def load_animals_list(animals_file: Path = ANIMALS_FILE) -> Set[str]:
    """Load the animals list from file and return as a set of lowercase names."""
    animals = set()
    
    with open(animals_file, 'r', encoding='utf-8') as f:
        for line in f:
            animal = line.strip()
            if animal:  # Skip empty lines
                animals.add(animal.lower())
    
    print(f"Loaded {len(animals)} animals from {animals_file}")
    return animals


def detect_animal_in_text(text: str, animals_set: Set[str]) -> str:
    """
    Detect which animal is mentioned in the text.
    Returns the animal name (original case) or empty string if none found.
    """
    if not text or not animals_set:
        return ""
    
    # Convert text to lowercase for matching
    text_lower = text.lower()
    
    # Sort animals by length (longest first) to prioritize longer matches
    # This helps with cases like "guinea pig" vs "pig"
    sorted_animals = sorted(animals_set, key=len, reverse=True)
    
    for animal in sorted_animals:
        # Use word boundaries to avoid partial matches
        # For multi-word animals, we need to be more flexible
        if ' ' in animal:
            # Multi-word animal - look for the phrase
            if animal in text_lower:
                # Check if it's not part of a larger word
                pattern = r'\b' + re.escape(animal) + r'\b'
                if re.search(pattern, text_lower):
                    return animal.title()  # Return with title case
        else:
            # Single word animal - use strict word boundaries
            pattern = r'\b' + re.escape(animal) + r'\b'
            if re.search(pattern, text_lower):
                return animal.title()  # Return with title case
    
    return ""


def load_model(model_name: str, gpus: Optional[int] = None):
    """Load a Hugging Face model and tokenizer for inference."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    print(f"✓ Successfully loaded {model_name}")
    return model, tokenizer


def clean_response(response: str, first_sentence: str = "") -> str:
    """Clean the generated response by removing extra formatting and repeated first sentence."""
    # Remove complete <think>...</think> tags and their content
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove incomplete/unclosed <think> tags
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
    
    # Remove any remaining </think> tags
    response = re.sub(r'</think>', '', response)
    
    # Strip whitespace
    response = response.strip()
    
    # Remove the first sentence if it's repeated in the continuation
    if first_sentence and response:
        # Clean up the first sentence for comparison (remove trailing punctuation)
        first_clean = first_sentence.rstrip('.!?').strip()
        
        # Check if response starts with the first sentence
        if response.lower().startswith(first_clean.lower()):
            # Remove the repeated first sentence
            response = response[len(first_clean):].strip()
            # Remove any leading punctuation that might be left
            response = response.lstrip('.!?').strip()
    
    return response


def generate_answers(
    message_lists: List[List[Dict[str, str]]],
    model,
    tokenizer,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[str]:
    """Generate answers using Hugging Face transformers."""
    
    # Convert message lists to prompts
    prompts = []
    for messages in message_lists:
        # Apply chat template if available, otherwise use simple format
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            prompt = messages[0]["content"]
        prompts.append(prompt)
    
    # Tokenize inputs
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    
    # Move to device
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode responses
    answers = []
    for i, output in enumerate(outputs):
        # Remove input tokens to get only the generated part
        input_length = inputs["input_ids"][i].shape[0]
        generated_tokens = output[input_length:]
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        cleaned_text = clean_response(generated_text)
        answers.append(cleaned_text)
    
    return answers


def load_tinystories_dataset() -> pd.DataFrame:
    """Load the TinyStories dataset from Hugging Face."""
    
    dataset = load_dataset("roneneldan/TinyStories")
    df = dataset["validation"].to_pandas()
        
    print(f"Loaded {len(df)} examples from TinyStories dataset")
    return df


def extract_first_sentence(text: str) -> str:
    """Extract the first sentence from a story."""
    if not text:
        return ""
    
    # Simple sentence splitting - look for sentence endings
    sentences = re.split(r'[.!?]+\s+', text.strip())
    
    if sentences:
        first_sentence = sentences[0].strip()
        # Add period if it doesn't end with punctuation
        if first_sentence and not first_sentence[-1] in '.!?':
            first_sentence += '.'
        return first_sentence
    
    return text.strip()


def create_continuation_prompt(first_sentence: str) -> List[Dict[str, str]]:
    """Create a prompt for the model to continue the story."""
    
    messages = [
        {
            "role": "user",
            "content": f"""/no_think Here’s the first sentence of a story: {first_sentence} Continue this story with one sentence that introduces a new animal character."""
        }
    ]
    
    return messages


def evaluate_model_on_dataset(
    model_name: str,
    dataset: pd.DataFrame,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gpus: Optional[int] = None,
    batch_size: int = 32,
    animals_set: Optional[Set[str]] = None
) -> List[Dict]:
    """Evaluate a single model on the entire dataset."""
    
    print(f"Evaluating {model_name} on {len(dataset)} examples...")
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name, gpus)
    
    results = []
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_data = dataset.iloc[batch_start:batch_end]
        
        # Create prompts for this batch
        message_lists = []
        first_sentences = []
        
        for _, row in batch_data.iterrows():
            story_text = row['text']
            first_sentence = extract_first_sentence(story_text)
            first_sentences.append(first_sentence)
            
            messages = create_continuation_prompt(first_sentence)
            message_lists.append(messages)
        
        # Generate answers for the batch
        generated_answers = generate_answers(message_lists, model, tokenizer, temperature, max_tokens)
        
        # Process results
        for i, (idx, row) in enumerate(batch_data.iterrows()):
            first_sentence = first_sentences[i]
            generated_answer = generated_answers[i]
            
            # Clean the generated answer to remove repeated first sentence
            cleaned_continuation = clean_response(generated_answer, first_sentence)
            
            # Detect animal in the generated continuation
            detected_animal = detect_animal_in_text(cleaned_continuation, animals_set)
            
            # Prepare result record
            result = {
                "model": model_name,
                "example_id": idx,
                "first_sentence": first_sentence,
                "generated_continuation": cleaned_continuation,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "detected_animal": detected_animal
            }
            
            results.append(result)
    
    # Clean up model to free memory
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    """Main evaluation loop."""
    # Hardcoded configuration from run_tinystories.sh
    SAMPLE_SIZE = 1000  # Use smaller sample for testing; remove for full dataset
    TEMPERATURE = 0.0
    MAX_TOKENS = 50
    BATCH_SIZE = 32
    RESULTS_DIR_PATH = Path("results/tinystories")
    
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
    
    print("")
    print("Configuration:")
    print(f"  Sample size: {SAMPLE_SIZE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("")
    
    # Create results directory
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_tinystories_dataset()
    original_size = len(dataset)
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    # Randomly sample the dataset
    if SAMPLE_SIZE < len(dataset):
        sample_indices = np.random.choice(len(dataset), size=SAMPLE_SIZE, replace=False)
        dataset = dataset.iloc[sample_indices].reset_index(drop=True)
        print(f"Randomly sampled {len(dataset)} examples from {original_size} total")
    else:
        print(f"Sample size ({SAMPLE_SIZE}) >= dataset size ({len(dataset)}), using full dataset")
    
    # Load animals list
    animals_set = load_animals_list()
    
    # Run evaluations for all models
    for i, model_name in enumerate(ALL_MODELS):
        print("")
        print("=" * 60)
        print(f"[{i+1}/{TOTAL_MODELS}] Starting evaluation for: {model_name}")
        print("=" * 60)
        
        # Evaluate model
        results = evaluate_model_on_dataset(
            model_name, dataset, TEMPERATURE, MAX_TOKENS, None, BATCH_SIZE, animals_set
        )
        
        # Save results as CSV
        model_name_clean = model_name.split("/")[-1]  # e.g., "Qwen3-0.6B"
        results_file = RESULTS_DIR_PATH / f"{model_name_clean}.csv"
        
        # Convert to DataFrame and save as CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)
        
        print(f"✓ Saved results for {model_name} to {results_file}")
        
        # Basic stats
        non_empty = len(results_df[results_df['generated_continuation'].str.len() > 0])
        print(f"Non-empty continuations: {non_empty}/{len(results_df)} ({non_empty/len(results_df):.2%})")
        
        # Show a few examples
        print(f"\nExample continuations:")
        for j, row in results_df.head(3).iterrows():
            print(f"  First sentence: {row['first_sentence']}")
            print(f"  Continuation: {row['generated_continuation']}")
            print(f"  Detected Animal: {row['detected_animal']}")
            print()
    
    # List generated CSV files
    print("")
    print("Generated results files:")
    csv_files = sorted(RESULTS_DIR_PATH.glob("*.csv"))
    for file in csv_files:
        print(f"  {file.name}")


if __name__ == "__main__":
    main() 