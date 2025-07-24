#!/usr/bin/env bash
# run_qwen3_models.sh
#
# Sequentially run sample_couplet_rhyme_distribution_vllm.py for all Qwen3 model sizes.
# Adjust INPUT_FILE, OUTPUT_DIR, and LOG_DIR as needed.

set -euo pipefail

cd /root/model-planning

INPUT_FILE="data/couplet_first_lines.txt"
OUTPUT_DIR="results/couplet_samples"

MODEL_SIZES=(0.6 1.7 4 8 14 32)

for SIZE in "${MODEL_SIZES[@]}"; do
  MODEL_ID="Qwen/Qwen3-${SIZE}B"
  echo "===== Running sampling for ${MODEL_ID} =====" >&2
  uv run sample_couplet_rhyme_distribution_vllm.py \
    --input "$INPUT_FILE" \
    --model "$MODEL_ID" \
    --output-dir "$OUTPUT_DIR" \
    --start-server \
    --port 8005
  echo "===== Completed ${MODEL_ID} =====" >&2
  echo >&2
done

echo "All Qwen3 models processed. Results in $OUTPUT_DIR and logs in $LOG_DIR" >&2 