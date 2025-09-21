#!/bin/bash

# Run TinyStories evaluation for all Qwen3 models and Llama-3-8B-Instruct
# This script runs the tinystories_eval_local.py script for multiple models
# with proper error handling.

# Configuration
SAMPLE_SIZE=1000  # Use smaller sample for testing; remove --sample-size for full dataset
TEMPERATURE=0.0
MAX_TOKENS=50
BATCH_SIZE=32
RESULTS_DIR="results/tinystories"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Define models to evaluate
QWEN3_MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

OTHER_MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Combine all models
ALL_MODELS=("${QWEN3_MODELS[@]}" "${OTHER_MODELS[@]}")

TOTAL_MODELS=${#ALL_MODELS[@]}

echo "Starting evaluation for $TOTAL_MODELS models:"
for i in "${!ALL_MODELS[@]}"; do
    echo "  $((i+1)). ${ALL_MODELS[$i]}"
done

echo ""
echo "Configuration:"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Temperature: $TEMPERATURE"
echo "  Max tokens: $MAX_TOKENS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Function to run evaluation for a single model
run_evaluation() {
    local model="$1"
    local model_num="$2"
    
    echo ""
    echo "============================================================"
    echo "[$model_num/$TOTAL_MODELS] Starting evaluation for: $model"
    echo "============================================================"
    
    # Build command
    cmd=(
        uv run tinystories_eval_huggingface.py
        --model "$model"
        --temperature "$TEMPERATURE"
        --max-tokens "$MAX_TOKENS"
        --batch-size "$BATCH_SIZE"
        --sample-size "$SAMPLE_SIZE"
    )
    
    # Run the evaluation
    "${cmd[@]}"
}

# Run evaluations for all models
for i in "${!ALL_MODELS[@]}"; do
    run_evaluation "${ALL_MODELS[$i]}" "$((i+1))"
done

# List generated CSV files
echo ""
echo "Generated results files:"
find "$RESULTS_DIR" -name "*.csv" -type f | sort | while read -r file; do
    echo "  $(basename "$file")"
done
