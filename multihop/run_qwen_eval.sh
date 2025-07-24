#!/bin/bash

# Define model sizes and their corresponding batch sizes
declare -A batch_sizes=(
    ["Qwen/Qwen3-0.6B"]="2048"
    ["Qwen/Qwen3-1.7B"]="1024"
    ["Qwen/Qwen3-4B"]="512"
    ["Qwen/Qwen3-8B"]="256"
    ["Qwen/Qwen3-14B"]="128"
    ["Qwen/Qwen3-32B"]="64"
)

# Define models in order from smallest to largest
models=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-14B"
    "Qwen/Qwen3-32B"
)

# Create results directory if it doesn't exist
mkdir -p results/behavioral

# Iterate over models in order and run evaluation
for model in "${models[@]}"; do
    batch_size="${batch_sizes[$model]}"
    echo "Evaluating $model with batch size $batch_size"
    
    python evaluate_multihop.py \
        --model "$model" \
        --batch_size "$batch_size" \
        --max_new_tokens 5 \
        --output_dir "results/behavioral"
    
    echo "Completed evaluation for $model"
    echo "----------------------------------------"
done

echo "All evaluations complete!" 