#!/usr/bin/env python3
"""
Evaluate all Qwen3 models on the TwoHopFact dataset using VLLM.

This script:
1. Loads the TwoHopFact dataset from Hugging Face
2. Sets up VLLM servers for each Qwen3 model
3. Evaluates each model's ability to extract relevant entities in multi-hop reasoning
4. Limits token generation to capture only the relevant entity
5. Saves results for analysis

Based on existing couplet evaluation patterns in this codebase.
"""

import argparse
import json
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm
import pandas as pd
import requests
from datasets import load_dataset
import numpy as np

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

DEFAULT_PORT = 8000
DEFAULT_TEMPERATURE = 0.0  # Low temperature for precise entity extraction
DEFAULT_MAX_TOKENS = 10    # Limit to just the entity tokens
RESULTS_DIR = Path("results/twohop_fact")

VLLM_BASE_CMD = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    "placeholder-model",
    "--dtype",
    "bfloat16",
    "--port",
    str(DEFAULT_PORT),
    "--trust-remote-code",
    "--max-model-len",
    "2048",  # Reasonable context for the task
]


def wait_for_port(host: str, port: int, timeout: float = 600.0) -> None:
    """Block until *host:port* accepts TCP connections or *timeout* seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            print("Waiting for server to become ready …")
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for server on {host}:{port}.")





def launch_vllm_server(model: str, port: int, gpus: Optional[int] = None) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server and return the subprocess handle."""
    cmd = VLLM_BASE_CMD.copy()
    cmd[cmd.index("--model") + 1] = model
    cmd[cmd.index("--port") + 1] = str(port)
    
    if gpus is not None:
        cmd.extend(["--tensor-parallel-size", str(gpus)])
    
    print("Starting vLLM server:\n  " + " ".join(cmd))
    return subprocess.Popen(cmd)


def generate_answer(
    prompt: str,
    model: str,
    port: int,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    """Generate answer using VLLM OpenAI-compatible API."""
    
    url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\n", ".", "?", "!", ","]  # Stop at common delimiters
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"].strip()
        else:
            return ""
            
    except Exception as e:
        print(f"Generation failed: {e}")
        return ""


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


def create_evaluation_prompt(example: Dict, mode: str = "2-hop") -> str:
    """Create a prompt for the model to answer the question."""
    
    # Select the appropriate column based on mode
    if mode == "1-hop":
        base_prompt = example['r1(e1).prompt']
    elif mode == "2-hop":
        base_prompt = example['r2(r1(e1)).prompt']
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be '1-hop' or '2-hop'")
    
    # Create a focused prompt that encourages concise answers
    evaluation_prompt = f"""Fact: {base_prompt}"""
    
    return evaluation_prompt


def evaluate_model_on_dataset(
    model: str,
    dataset: pd.DataFrame,
    port: int,
    mode: str = "2-hop",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[Dict]:
    """Evaluate a single model on the entire dataset."""
    
    print(f"Evaluating {model} on {len(dataset)} examples in {mode} mode...")
    
    # Determine the correct answer column based on mode
    answer_column = "e2.value" if mode == "1-hop" else "e3.value"
    
    results = []
    
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        # Create evaluation prompt
        prompt = create_evaluation_prompt(row.to_dict(), mode)
        
        # Generate answer
        generated_answer = generate_answer(
            prompt, model, port, temperature, max_tokens
        )
        
        # Get ground truth answer
        ground_truth_answer = row[answer_column]
        
        # Prepare result record
        result = {
            "model": model,
            "mode": mode,
            "example_id": idx,
            "prompt": prompt,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add original data for reference
        for col in row.index:
            result[f"original_{col}"] = row[col]
        
        results.append(result)
    
    return results


def main():
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 models on TwoHopFact dataset"
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
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Base port for VLLM servers"
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
        
        current_port = args.port + model_idx
        server_process = None
        
        # Launch server and ensure graceful cleanup on interrupt
        server_process = launch_vllm_server(model, current_port, args.gpus)
        signal.signal(signal.SIGINT, lambda *_: server_process.terminate() if server_process else None)
        
        try:
            try:
                wait_for_port("localhost", current_port)
                print("Server port is ready, allowing extra time for model loading...")
                time.sleep(10)  # Give server extra time to fully initialize
                
                # Test the server with a simple request
                print(f"Testing server readiness with a simple request on port {current_port}...")
                test_response = generate_answer("Hello", model, current_port, 0.0, 1)
                print(f"✓ Server test successful: '{test_response}'")
                
            except RuntimeError as exc:
                print(f"VLLM server failed to start: {exc}")
                continue
            
            # Evaluate model
            model_results = evaluate_model_on_dataset(
                model, dataset, current_port, args.mode, args.temperature, args.max_tokens
            )
            
            all_results.extend(model_results)
            
            # Save individual model results
            model_name = model.split("/")[-1]  # e.g., "Qwen3-14B"
            model_results_file = args.results_dir / f"{model_name}_{args.mode}_results.json"
            
            with open(model_results_file, "w") as f:
                json.dump(model_results, f, indent=2)
            
            print(f"✓ Saved results for {model} to {model_results_file}")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
            
        except Exception as e:
            print(f"✗ Error evaluating {model}: {e}")
            
        finally:
            # Clean up server
            if server_process:
                print(f"Stopping VLLM server for {model}...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server_process.kill()
    
    # Save combined results
    if all_results:
        combined_file = args.results_dir / f"combined_{args.mode}_results.json"
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Also save as CSV for easier analysis
        results_df = pd.DataFrame(all_results)
        csv_file = args.results_dir / f"combined_{args.mode}_results.csv"
        results_df.to_csv(csv_file, index=False)
        
        print(f"\n✓ Saved combined results to {combined_file} and {csv_file}")
        
        # Print summary
        print(f"\nEvaluation Summary ({args.mode}):")
        print(f"Total examples processed: {len(results_df)}")
        print(f"Models evaluated: {results_df['model'].nunique()}")
        print(f"Results saved to: {args.results_dir}")
        
        # Basic accuracy if ground truth is available
        if "ground_truth_answer" in results_df.columns:
            print("\nModel performance preview:")
            for model in results_df['model'].unique():
                model_data = results_df[results_df['model'] == model]
                non_empty = len(model_data[model_data['generated_answer'].str.len() > 0])
                
                # Simple exact match accuracy
                exact_matches = len(model_data[model_data['generated_answer'] == model_data['ground_truth_answer']])
                accuracy = exact_matches / len(model_data) if len(model_data) > 0 else 0
                
                print(f"  {model}: {non_empty}/{len(model_data)} non-empty, {exact_matches}/{len(model_data)} exact matches ({accuracy:.2%})")


if __name__ == "__main__":
    main()
