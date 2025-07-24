#!/bin/bash

# Define model sizes and their corresponding batch sizes
declare -A batch_sizes=(
    ["Qwen/Qwen3-0.6B-Base"]="2048"
    ["Qwen/Qwen3-1.7B-Base"]="1024"
    ["Qwen/Qwen3-4B-Base"]="512"
    ["Qwen/Qwen3-8B-Base"]="256"
    ["Qwen/Qwen3-14B-Base"]="128"
)

# Define models in order from smallest to largest
models=(
    "Qwen/Qwen3-0.6B-Base"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B-Base"
    "Qwen/Qwen3-14B-Base"
)

# Create results directory if it doesn't exist
mkdir -p results/behavioral-base-continuation

# Iterate over models in order and run evaluation
for model in "${models[@]}"; do
    batch_size="${batch_sizes[$model]}"
    echo "Evaluating $model with batch size $batch_size"
    
    python evaluate_multihop_base.py \
        --model "$model" \
        --batch_size "$batch_size" \
        --max_new_tokens 5 \
        --output_dir "results/behavioral-base-continuation"
    
    echo "Completed evaluation for $model"
    echo "----------------------------------------"
done

echo "All evaluations complete!" 