#!/usr/bin/env python3
"""
Evaluate all Qwen3 models on the TwoHopFact dataset using local VLLM inference.

This script:
1. Loads the TwoHopFact dataset from Hugging Face
2. Uses local VLLM inference for each Qwen3 model
3. Evaluates each model's ability to extract relevant entities in multi-hop reasoning
4. Limits token generation to capture only the relevant entity
5. Saves results for analysis

Based on existing couplet evaluation patterns in this codebase.
"""

import argparse
import json
import gc
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import numpy as np
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Available Qwen3 model sizes based on codebase patterns
QWEN3_MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B", 
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B"
]

DEFAULT_TEMPERATURE = 0.0  # Low temperature for precise entity extraction
DEFAULT_MAX_TOKENS = 20    # Limit to just the entity tokens
RESULTS_DIR = Path("results/twohop_fact")


def load_model(model_name: str, gpus: Optional[int] = None) -> LLM:
    """Load a VLLM model for local inference."""
    print(f"Loading model: {model_name}")
    
    # Set up model parameters
    model_kwargs = {
        "model": model_name,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": 2048,  # Reasonable context for the task
        "gpu_memory_utilization": 0.9,
        "swap_space": 4,  # GB of swap space for CPU offloading
    }
    
    # Add tensor parallel if specified
    if gpus is not None and gpus > 1:
        model_kwargs["tensor_parallel_size"] = gpus
    
    try:
        llm = LLM(**model_kwargs)
        print(f"✓ Successfully loaded {model_name}")
        return llm
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")
        raise


def clean_response(response: str) -> str:
    """Clean the generated response by removing think tags and extra punctuation."""
    # Remove complete <think>...</think> tags and their content
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove incomplete/unclosed <think> tags
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
    
    # Remove any remaining </think> tags
    response = re.sub(r'</think>', '', response)
    
    # Strip whitespace and common trailing punctuation
    response = response.strip()
    response = response.rstrip('.,!?;')
    response = response.strip()
    
    # If the response is empty after cleaning, return empty string
    if not response:
        return ""
    
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


def load_twohop_dataset() -> pd.DataFrame:
    """Load the TwoHopFact dataset from Hugging Face."""
    print("Loading TwoHopFact dataset...")
    
    # Load the dataset
    dataset = load_dataset("soheeyang/TwoHopFact")
    
    # Convert to pandas DataFrame for easier manipulation
    if "test" in dataset:
        df = dataset["test"].to_pandas()
    elif "validation" in dataset:
        df = dataset["validation"].to_pandas()
    else:
        # Use train split if no test/validation
        df = dataset["train"].to_pandas()
        
    print(f"Loaded {len(df)} examples from TwoHopFact dataset")
    return df


def create_evaluation_prompt(example: Dict, mode: str = "2-hop") -> List[Dict[str, str]]:
    """Create a prompt for the model to answer the question."""
    
    # Select the appropriate column based on mode
    if mode == "1-hop":
        base_prompt = example['r1(e1).prompt']
    elif mode == "2-hop":
        base_prompt = example['r2(r1(e1)).prompt']
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be '1-hop' or '2-hop'")
    
    # Create a focused prompt that encourages concise answers
    messages = [
        {
            "role": "user",
            "content": f"/no_think Finish this sentence, saying only the answer: {base_prompt}"
        }
    ]
    
    return messages


def evaluate_model_on_dataset(
    model: str,
    dataset: pd.DataFrame,
    mode: str = "2-hop",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gpus: Optional[int] = None,
    batch_size: int = 32
) -> List[Dict]:
    """Evaluate a single model on the entire dataset."""
    
    print(f"Evaluating {model} on {len(dataset)} examples in {mode} mode...")
    
    # Load the model
    llm = load_model(model, gpus)
    
    # Determine the correct answer column based on mode
    answer_column = "e2.value" if mode == "1-hop" else "e3.value"
    
    results = []
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch_data = dataset.iloc[batch_start:batch_end]
        
        # Create prompts for this batch
        message_lists = []
        for _, row in batch_data.iterrows():
            messages = create_evaluation_prompt(row.to_dict(), mode)
            message_lists.append(messages)
        
        # Generate answers for the batch
        generated_answers = generate_answers(message_lists, llm, temperature, max_tokens)
        
        # Process results
        for i, (idx, row) in enumerate(batch_data.iterrows()):
            messages = message_lists[i]
            # Store the user prompt content for logging
            prompt_content = messages[0]["content"]
            generated_answer = generated_answers[i]
            ground_truth_answer = row[answer_column]
            
            # Prepare result record
            result = {
                "model": model,
                "mode": mode,
                "example_id": idx,
                "prompt": prompt_content,
                "generated_answer": generated_answer,
                "ground_truth_answer": ground_truth_answer,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add original data for reference
            for col in row.index:
                result[f"original_{col}"] = row[col]
            
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
        description="Evaluate Qwen3 models on TwoHopFact dataset using local VLLM"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=QWEN3_MODELS,
        help="List of models to evaluate"
    )
    parser.add_argument(
        "--mode",
        choices=["1-hop", "2-hop"],
        default="2-hop",
        help="Evaluation mode: 1-hop or 2-hop reasoning"
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
    dataset = load_twohop_dataset()
    original_size = len(dataset)
    
    if args.sample_size:
        # Set random seed for reproducible sampling across models
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
    
    # Evaluate each model
    all_results = []
    
    for model_idx, model in enumerate(args.models):
        print(f"\n{'='*50}")
        print(f"Evaluating model {model_idx + 1}/{len(args.models)}: {model} ({args.mode})")
        print(f"{'='*50}")
        
        try:
            # Evaluate model
            model_results = evaluate_model_on_dataset(
                model, dataset, args.mode, args.temperature, args.max_tokens, args.gpus, args.batch_size
            )
            
            all_results.extend(model_results)
            
            # Save individual model results as CSV
            model_name = model.split("/")[-1]  # e.g., "Qwen3-14B"
            model_results_file = args.results_dir / f"{model_name}_{args.mode}_results.csv"
            
            # Convert to DataFrame and save as CSV
            model_df = pd.DataFrame(model_results)
            model_df.to_csv(model_results_file, index=False)
            
            print(f"✓ Saved results for {model} to {model_results_file}")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
            
        except Exception as e:
            print(f"✗ Error evaluating {model}: {e}")
            # Continue with next model
            continue
    
    # Print summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\nEvaluation Summary ({args.mode}):")
        print(f"Total examples processed: {len(results_df)}")
        print(f"Models evaluated: {results_df['model'].nunique()}")
        print(f"Individual model results saved to: {args.results_dir}")
        
        # Basic accuracy if ground truth is available
        if "ground_truth_answer" in results_df.columns:
            print("\nModel performance preview:")
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                non_empty = len(model_data[model_data['generated_answer'].str.len() > 0])
                
                # Substring match accuracy (ground truth is substring of generated answer)
                correct_matches = 0
                for _, row in model_data.iterrows():
                    gt_answer = str(row['ground_truth_answer']).strip()
                    gen_answer = str(row['generated_answer']).strip()
                    if gt_answer and gen_answer and gt_answer != 'nan' and gen_answer != 'nan':
                        if gt_answer.lower() in gen_answer.lower():
                            correct_matches += 1
                
                accuracy = correct_matches / len(model_data) if len(model_data) > 0 else 0
                
                print(f"  {model}: {non_empty}/{len(model_data)} non-empty, {correct_matches}/{len(model_data)} correct ({accuracy:.2%})")
    
    else:
        print("\nNo results to save - all models failed to evaluate")


if __name__ == "__main__":
    main() 