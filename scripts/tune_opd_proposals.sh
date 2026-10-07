#!/usr/bin/env bash
# All pretrained models by default; execution keys deduplicate benchmarks.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
models="${OPD_TUNE_MODELS:-qwen25_1p5b,qwen25_3b,qwen25_7b,qwen25_14b,qwen3_1p7b,qwen3_4b}"
cmd=("$PYTHON_BIN" "$ROOT/scripts/tune_opd_proposals.py"
  --models "$models" --pretrain-root "${PRETRAIN_ROOT:-$OUTPUT_ROOT/pretrain}"
  --profile-dir "${OPD_PROPOSAL_PROFILE_DIR:-$OUTPUT_ROOT/benchmarks/opd_proposals}"
  --dtype "${OPD_TUNE_DTYPE:-${MODEL_DTYPE:-bf16}}"
  --topk "${OPD_TOPK:-16}" --iterations "${OPD_TUNE_ITERATIONS:-30}"
  --batch-size "${BATCH_SIZE:-8}" --responses "${RESPONSES_PER_PROMPT:-${REPEATED_GENERATE_NUMS:-8}}"
  --max-draft-k "${MAX_DRAFT_K:-8}"
  --context-points "${OPD_TUNE_CONTEXT_POINTS:-7}" --active-points "${OPD_TUNE_ACTIVE_POINTS:-8}")
if [[ -n "${OPD_RANK:-}" ]];then cmd+=(--rank "$OPD_RANK");fi
if [[ -n "${OPD_TUNE_SHAPES:-}" ]];then cmd+=(--shapes "$OPD_TUNE_SHAPES");fi
if [[ -n "${OPD_TUNE_SLOTS:-}" ]];then cmd+=(--slots "$OPD_TUNE_SLOTS");fi
if [[ -n "${OPD_TUNE_OUTPUT:-}" ]];then cmd+=(--output "$OPD_TUNE_OUTPUT");fi
if [[ "$models" != *,* ]];then
  for pair in DRAFT_CONFIG:draft-config DRAFT_CHECKPOINT:draft-checkpoint VOCAB_MAPPING:vocab-mapping;do
    name="${pair%%:*}";flag="${pair#*:}"
    if [[ -n "${!name:-}" ]];then cmd+=("--$flag" "${!name}");fi
  done
  if [[ -n "${PRETRAIN_MODEL_ROOT:-}" ]];then
    cmd+=(--draft-config "${DRAFT_CONFIG:-$PRETRAIN_MODEL_ROOT/latest_draft_config.json}"
      --draft-checkpoint "${DRAFT_CHECKPOINT:-$PRETRAIN_MODEL_ROOT/latest_checkpoint}"
      --vocab-mapping "${VOCAB_MAPPING:-$PRETRAIN_MODEL_ROOT/latest_vocab_mapping.pt}")
  fi
fi
exec "${cmd[@]}" "$@"
