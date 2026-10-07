#!/usr/bin/env bash
set -euo pipefail
export METHOD=fastgrpo
export DATASET="${DATASET:-simplelr}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TARGET_LR="${TARGET_LR:-1e-5}"
export DRAFT_LR="${DRAFT_LR:-1e-5}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"

# Backward-compatible default: train Qwen2.5-7B FastGRPO.
MODEL_KEY="${MODEL_KEY:-qwen25_7b}"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-7B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"
