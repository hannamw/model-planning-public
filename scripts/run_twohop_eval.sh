#!/bin/bash

# Two-Hop Fact Evaluation Script
# Runs evaluation on both 1-hop and 2-hop modes with 10000 examples for all Qwen3 models

set -e  # Exit on any error

echo "Starting Two-Hop Fact Evaluation..."
echo "Models: All Qwen3 models (0.6B, 1.7B, 4B, 8B, 14B, 32B)"
echo "Sample size: 10000 examples"
echo "Modes: 1-hop and 2-hop"
echo ""

# Create results directory if it doesn't exist
mkdir -p results/twohop_fact

# Run 1-hop evaluation
echo "=========================================="
echo "Running 1-hop evaluation..."
echo "=========================================="
uv run two_hop_eval_local.py \
    --mode 1-hop \
    --sample-size 10000 \
    --batch-size 32 \
    --temperature 0.0 \
    --max-tokens 20

echo ""
echo "1-hop evaluation completed!"
echo ""

# Run 2-hop evaluation  
echo "=========================================="
echo "Running 2-hop evaluation..."
echo "=========================================="
uv run two_hop_eval_local.py \
    --mode 2-hop \
    --sample-size 10000 \
    --batch-size 32 \
    --temperature 0.0 \
    --max-tokens 20

echo ""
echo "2-hop evaluation completed!"
echo ""

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in: results/twohop_fact/"
echo "=========================================="

# List the result files
echo "Generated files:"
ls -la results/twohop_fact/

echo ""
echo "Evaluation complete!" 