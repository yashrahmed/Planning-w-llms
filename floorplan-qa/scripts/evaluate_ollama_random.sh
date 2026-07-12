#!/usr/bin/env bash

set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [seed] [output.json]" >&2
  exit 2
fi

SEED="${1:-0}"
MODEL="${MODEL:-qwen3.5:4b}"
SAMPLE_SIZE="${SAMPLE_SIZE:-150}"
POOL_LAYOUTS="${POOL_LAYOUTS:-80}"
MAX_TOKENS="${MAX_TOKENS:-768}"

if [[ ! ${SEED} =~ ^-?[0-9]+$ ]]; then
  echo "Error: seed must be an integer." >&2
  exit 2
fi

for value_name in SAMPLE_SIZE POOL_LAYOUTS MAX_TOKENS; do
  value="${!value_name}"
  if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: ${value_name} must be a positive integer." >&2
    exit 2
  fi
done

if (( SAMPLE_SIZE > POOL_LAYOUTS * 8 )); then
  echo "Error: SAMPLE_SIZE exceeds the generated question pool." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
POOL_DIR="${PROJECT_DIR}/datasets/ollama-eval-pools/seed-${SEED}"
OUTPUT_PATH="${2:-${PROJECT_DIR}/datasets/evaluations/qwen3.5-4b-seed-${SEED}.json}"

cd "${PROJECT_DIR}"

echo "Generating ${POOL_LAYOUTS}-layout evaluation pool with seed ${SEED} ..."
uv run python -m floorplan_qa.generate_questions \
  --num-layouts "${POOL_LAYOUTS}" \
  --seed "${SEED}" \
  --output-dir "${POOL_DIR}"

echo "Evaluating a uniform ${SAMPLE_SIZE}-question sample with ${MODEL} ..."
uv run python -m floorplan_qa.evaluate_jsonl ollama \
  "${POOL_DIR}/questions.jsonl" \
  --model "${MODEL}" \
  --seed "${SEED}" \
  --sample-size "${SAMPLE_SIZE}" \
  --max-tokens "${MAX_TOKENS}" \
  --output-json "${OUTPUT_PATH}"
