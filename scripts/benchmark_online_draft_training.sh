#!/usr/bin/env bash
# Opt-in real B200 comparison: identical target/draft initialization and seed.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-3}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$REPO_DIR/outputs/benchmarks/online_draft_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ "$BENCHMARK_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: BENCHMARK_STEPS must be positive" >&2; exit 2; }
[[ ! -e "$BENCHMARK_ROOT" ]] || { echo "ERROR: use a new BENCHMARK_ROOT" >&2; exit 2; }
export MODEL_KEY BATCH_SIZE="${BATCH_SIZE:-8}" RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-${REPEATED_GENERATE_NUMS:-8}}"
export TARGET_LR="${TARGET_LR:-1e-6}" DRAFT_LR="${DRAFT_LR:-1e-4}" ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
export DATASET="${DATASET:-dapo}" RESUME="" REFLEX_PROFILE=0 REFLEX_DIAGNOSTICS=0
for draft_mode in per_response batched; do
  env METHOD=fastgrpo DRAFT_TRAIN_MODE="$draft_mode" RUN_NAME="draft_${draft_mode}" \
    RUN_DIR="$BENCHMARK_ROOT/$draft_mode" TRAIN_MODEL_ROOT="$BENCHMARK_ROOT/${draft_mode}_links" \
    bash "$REPO_DIR/scripts/launch/train_model.sh" --max_grpo_steps "$BENCHMARK_STEPS" "$@"
done
if [[ "${DRY_RUN:-false}" == "true" ]]; then exit 0; fi
"${PYTHON_BIN:-python3}" "$REPO_DIR/scripts/summarize_online_draft_training.py" "$BENCHMARK_ROOT" \
  | tee "$BENCHMARK_ROOT/benchmark_report.json"
