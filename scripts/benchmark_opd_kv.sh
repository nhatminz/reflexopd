#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_KEY="${MODEL_KEY:-qwen3_1p7b}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
source "$ROOT/configs/_shared/b200_common.env"
source "$ROOT/configs/$MODEL_KEY/b200.env"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$PRETRAIN_MODEL_ROOT/latest_draft_config.json}"
KV_BENCH_OUTPUT="${KV_BENCH_OUTPUT:-$OUTPUT_ROOT/benchmarks/kv_${MODEL_KEY}_$(date -u +%Y%m%dT%H%M%S_%N)}"
[[ -f "$MODEL/config.json" ]] || { echo "Missing target config: $MODEL/config.json" >&2; exit 2; }
[[ -f "$DRAFT_CONFIG" ]] || { echo "Missing draft config: $DRAFT_CONFIG" >&2; exit 2; }
cmd=("$PYTHON_BIN" "$ROOT/scripts/benchmark_opd_kv.py"
  --lengths "${KV_BENCH_LENGTHS:-256,512,1024,2048}" --batches "${KV_BENCH_BATCHES:-16,32,64}"
  --dtype "$MODEL_DTYPE" --iterations "${KV_BENCH_ITERATIONS:-20}"
  --max-draft-length "$MAX_DRAFT_TOKEN_LENGTH" --max-draft-k "$MAX_DRAFT_K"
  --verification-capacity "$MAX_VERIFICATION_NUM")
if [[ -n "${KV_BENCH_LAYERS:-}" ]]; then cmd+=(--layers "$KV_BENCH_LAYERS"); fi
"${cmd[@]}" --config "$MODEL/config.json" --output "$KV_BENCH_OUTPUT/target" "$@"
"${cmd[@]}" --config "$DRAFT_CONFIG" --output "$KV_BENCH_OUTPUT/draft" "$@"
printf 'KV reports: %s/{target,draft}/{report.json,kv.csv}\n' "$KV_BENCH_OUTPUT"
