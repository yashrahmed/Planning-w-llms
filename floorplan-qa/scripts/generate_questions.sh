#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <number-of-examples>" >&2
  exit 2
fi

if [[ ! $1 =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: number-of-examples must be a positive integer." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"
uv run python -m floorplan_qa.generate_questions --num-examples "$1"
