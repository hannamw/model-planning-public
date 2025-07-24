#!/usr/bin/env bash
# run_twohop_evaluation.sh
#
# Evaluate all Qwen3 models on the TwoHopFact dataset using VLLM.
# This script follows the patterns from existing evaluation scripts in the repo.

set -euo pipefail

cd /root/model-planning

# Configuration
RESULTS_DIR="results/twohop_fact"
PYTHON_SCRIPT="two_hop_eval.py"
BASE_PORT=8005

# Qwen3 model sizes to evaluate
MODEL_SIZES=(0.6 1.7 4 8 14 32)

# Evaluation parameters
MODE="2-hop"  # Choose between "1-hop" or "2-hop"
TEMPERATURE=0.0
MAX_TOKENS=10
SAMPLE_SIZE=100  # Set to empty string "" to use full dataset

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "===== TwoHopFact Evaluation Script =====" >&2
echo "Mode: $MODE" >&2
echo "Results directory: $RESULTS_DIR" >&2
echo "Using temperature: $TEMPERATURE" >&2
echo "Max tokens: $MAX_TOKENS" >&2
if [ -n "$SAMPLE_SIZE" ]; then
  echo "Sample size: $SAMPLE_SIZE examples" >&2
else
  echo "Sample size: full dataset" >&2
fi
echo "Starting port: $BASE_PORT" >&2
echo >&2

# Build model list
MODELS=()
for SIZE in "${MODEL_SIZES[@]}"; do
  MODELS+=("Qwen/Qwen3-${SIZE}B")
done

echo "Models to evaluate:" >&2
for MODEL in "${MODELS[@]}"; do
  echo "  - $MODEL" >&2
done
echo >&2

# Run the evaluation
echo "Starting evaluation..." >&2

# Build command with conditional sample-size argument
CMD_ARGS=(
  --models "${MODELS[@]}"
  --mode "$MODE"
  --port "$BASE_PORT"
  --temperature "$TEMPERATURE"
  --max-tokens "$MAX_TOKENS"
  --results-dir "$RESULTS_DIR"
)

# Add sample-size only if specified
if [ -n "$SAMPLE_SIZE" ]; then
  CMD_ARGS+=(--sample-size "$SAMPLE_SIZE")
fi

uv run "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

echo >&2
echo "===== Evaluation Complete =====" >&2
echo "Results saved to: $RESULTS_DIR" >&2

# Show a summary of results
if [ -f "$RESULTS_DIR/combined_${MODE}_results.csv" ]; then
  echo >&2
  echo "Results summary:" >&2
  echo "$(wc -l < "$RESULTS_DIR/combined_${MODE}_results.csv") total result rows" >&2
  
  # Count results per model if possible
  if command -v csvcut >/dev/null 2>&1 && command -v csvstat >/dev/null 2>&1; then
    echo >&2
    echo "Results per model:" >&2
    csvcut -c model "$RESULTS_DIR/combined_${MODE}_results.csv" | tail -n +2 | sort | uniq -c >&2
  fi
fi

echo >&2
echo "To analyze results, see:" >&2
echo "  - $RESULTS_DIR/combined_${MODE}_results.csv" >&2
echo "  - $RESULTS_DIR/combined_${MODE}_results.json" >&2
echo "  - Individual model results: $RESULTS_DIR/*_${MODE}_results.json" >&2 