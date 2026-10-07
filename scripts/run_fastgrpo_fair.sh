#!/usr/bin/env bash
set -euo pipefail

# Fair in-repository baseline: identical SpecNaacl runtime with Reflex disabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHOD="fastgrpo"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
export DATASET="${DATASET:-simplelr}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TARGET_LR="${TARGET_LR:-1e-6}"
export DRAFT_LR="${DRAFT_LR:-1e-4}"
export OPD_FAST_LR="${OPD_FAST_LR:-${FAST_LR:-0.01}}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
source "$SCRIPT_DIR/launch/train_model.sh" "$@"
