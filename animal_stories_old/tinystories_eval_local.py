#!/usr/bin/env python3
"""
Evaluate a single model on TinyStories dataset using local VLLM inference.

This script:
1. Loads the TinyStories dataset from Hugging Face
2. Extracts the first sentence from each story
3. Prompts the model to generate one more sentence (continuation)
4. Uses local VLLM inference
5. Allows sampling a subset of examples
6. Saves results for analysis
7. Detects which animal is mentioned in the continuation
"""

import argparse
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
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 50    # Enough for one sentence
RESULTS_DIR = Path("results/tinystories")
ANIMALS_FILE = Path("data/animals.txt")


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


def load_model(model_name: str, gpus: Optional[int] = None) -> LLM:
    """Load a VLLM model for local inference."""
    print(f"Loading model: {model_name}")
    
    # Set up model parameters
    model_kwargs = {
        "model": model_name,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.9,
        "swap_space": 4,  # GB of swap space for CPU offloading
    }
    
    # Add tensor parallel if specified
    if gpus is not None and gpus > 1:
        model_kwargs["tensor_parallel_size"] = gpus    

    llm = LLM(**model_kwargs)
    print(f"✓ Successfully loaded {model_name}")
    return llm


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
    llm: LLM,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[str]:
    """Generate answers using local VLLM inference."""
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    
    # Generate responses using properly formatted prompts
    outputs = llm.chat(message_lists, sampling_params)
    
    # Extract and clean generated text
    answers = []
    for output in outputs:
        if output.outputs and len(output.outputs) > 0:
            generated_text = output.outputs[0].text.strip()
            cleaned_text = clean_response(generated_text)
            answers.append(cleaned_text)
        else:
            answers.append("")
    
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
    model: str,
    dataset: pd.DataFrame,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gpus: Optional[int] = None,
    batch_size: int = 32,
    animals_set: Optional[Set[str]] = None
) -> List[Dict]:
    """Evaluate a single model on the entire dataset."""
    
    print(f"Evaluating {model} on {len(dataset)} examples...")
    
    # Load the model
    llm = load_model(model, gpus)
    
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
        generated_answers = generate_answers(message_lists, llm, temperature, max_tokens)
        
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
                "model": model,
                "example_id": idx,
                "first_sentence": first_sentence,
                "generated_continuation": cleaned_continuation,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "detected_animal": detected_animal
            }
            
            results.append(result)
    
    # Clean up model to free memory
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(
        description="Evaluate a single model on TinyStories dataset using local VLLM"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="Number of GPUs for tensor parallel"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory to save results"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Limit dataset to N examples for testing"
    )
    
    args = parser.parse_args()
    
    # Create results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_tinystories_dataset()
    original_size = len(dataset)
    
    if args.sample_size:
        # Set random seed for reproducible sampling
        np.random.seed(42)
        
        # Randomly sample the dataset
        if args.sample_size < len(dataset):
            sample_indices = np.random.choice(len(dataset), size=args.sample_size, replace=False)
            dataset = dataset.iloc[sample_indices].reset_index(drop=True)
            print(f"Randomly sampled {len(dataset)} examples from {original_size} total")
        else:
            print(f"Sample size ({args.sample_size}) >= dataset size ({len(dataset)}), using full dataset")
    else:
        print(f"Using full dataset with {len(dataset)} examples")
    
    # Load animals list
    animals_set = load_animals_list()
    
    # Evaluate the model
    print(f"\n{'='*50}")
    print(f"Evaluating model: {args.model}")
    print(f"{'='*50}")
    

    # Evaluate model
    results = evaluate_model_on_dataset(
        args.model, dataset, args.temperature, args.max_tokens, args.gpus, args.batch_size, animals_set
    )
    
    # Save results as CSV
    model_name = args.model.split("/")[-1]  # e.g., "Qwen3-0.6B"
    results_file = args.results_dir / f"{model_name}_tinystories_results.csv"
    
    # Convert to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    print(f"✓ Saved results for {args.model} to {results_file}")
        # Basic stats
    non_empty = len(results_df[results_df['generated_continuation'].str.len() > 0])
    print(f"Non-empty continuations: {non_empty}/{len(results_df)} ({non_empty/len(results_df):.2%})")
    
    # Show a few examples
    print(f"\nExample continuations:")
    for i, row in results_df.head(3).iterrows():
        print(f"  First sentence: {row['first_sentence']}")
        print(f"  Continuation: {row['generated_continuation']}")
        print(f"  Detected Animal: {row['detected_animal']}")
        print()
    


if __name__ == "__main__":
    main() 