#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Edit only this list, or pass run directories as command-line arguments.
RUN_DIRS=(
  # "/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/run1"
  # "/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/run2"
)
if (($#)); then RUN_DIRS=("$@"); fi
if ((${#RUN_DIRS[@]} == 0)); then
  echo "ERROR: add paths to RUN_DIRS or pass run directories as arguments" >&2
  exit 2
fi
"$PYTHON_BIN" "$SCRIPT_DIR/plot_training_time.py" "${RUN_DIRS[@]}" \
  --output "${PLOT_OUTPUT:-training_time.png}"
