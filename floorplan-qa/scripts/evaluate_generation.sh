#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <questions.jsonl> [comparison-questions.jsonl]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

ARGS=("$1")
if [[ $# -eq 2 ]]; then
  ARGS+=(--compare "$2")
fi

uv run python -m floorplan_qa.quality_metrics "${ARGS[@]}"
