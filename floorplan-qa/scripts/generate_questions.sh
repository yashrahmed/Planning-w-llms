#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <number-of-layouts> [seed] [train|test|val]" >&2
  exit 2
fi

SEED="${2:-0}"
SPLIT="${3:-train}"

if [[ ! ${SEED} =~ ^-?[0-9]+$ ]]; then
  echo "Error: seed must be an integer." >&2
  exit 2
fi

if [[ ! $1 =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: number-of-examples must be a positive integer." >&2
  exit 2
fi

if [[ ! ${SPLIT} =~ ^(train|test|val)$ ]]; then
  echo "Error: split must be train, test, or val." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
uv run python -m floorplan_qa.generate_questions \
  --num-layouts "$1" \
  --seed "${SEED}" \
  --split "${SPLIT}"
