#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <number-of-examples> [seed]" >&2
  exit 2
fi

SEED="${2:-0}"

if [[ ! ${SEED} =~ ^-?[0-9]+$ ]]; then
  echo "Error: seed must be an integer." >&2
  exit 2
fi

if [[ ! $1 =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: number-of-examples must be a positive integer." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
uv run python -m floorplan_qa.generate_questions \
  --num-examples "$1" \
  --seed "${SEED}"
