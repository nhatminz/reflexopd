#!/usr/bin/env bash
set -euo pipefail
MODEL_KEY="qwen3_1p7b"; MODEL="${MODEL:-/workspace/storage-shared/models/Qwen3-1.7B}"
PRETRAIN_DATASET="${PRETRAIN_DATASET:-sharegpt}"; PRETRAIN_LR="${PRETRAIN_LR:-5e-5}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-16}"; PRETRAIN_MAX_LENGTH="${PRETRAIN_MAX_LENGTH:-2048}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/pretrain_model.sh"
