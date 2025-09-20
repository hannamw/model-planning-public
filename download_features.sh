#!/bin/bash

# Create features directory if it doesn't exist
mkdir -p features
cd features

# Define the list of (hf-repo-name, model-name) pairs

models=(
    "mwhanna/qwen3-0.6b-transcoders-lowl0,Qwen3-0.6B"
    "mwhanna/qwen3-1.7b-transcoders-lowl0,Qwen3-1.7B" 
    "mwhanna/qwen3-4b-transcoders,Qwen3-4B"
    "mwhanna/qwen3-8b-transcoders,Qwen3-8B"
    "mwhanna/qwen3-14b-transcoder-lowl0,Qwen3-14B"
)

# Iterate over each model pair
for model_pair in "${models[@]}"; do
    # Split the pair into repo name and model name
    IFS=',' read -r hf_repo_name model_name <<< "$model_pair"
    
    echo "Downloading features for $model_name from $hf_repo_name..."
    
    # Run the huggingface-cli download command
    uv run huggingface-cli download "$hf_repo_name" --include "features/*" --local-dir "$model_name" --local-dir-use-symlinks False
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully downloaded features for $model_name"
    else
        echo "✗ Failed to download features for $model_name"
    fi
    echo ""
done

echo "Feature download process completed."
