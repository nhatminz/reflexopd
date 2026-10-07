#!/usr/bin/env bash
# Generic real-SpecForge EAGLE-3 offline pretraining launcher.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${COMMON_ENV:-$PROJECT_DIR/configs/_shared/b200_common.env}"
source "${MODEL_ENV:-$PROJECT_DIR/configs/$MODEL_KEY/b200.env}"
: "${MODEL:?MODEL is required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"

case "${PRETRAIN_DATASET,,}" in
  sharegpt)
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
    ;;
  gsm8k)
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}"
    ;;
  simplelr|simplelr_abel|simplelr_abel_level3to5)
    PRETRAIN_DATASET="simplelr"
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}"
    ;;
  dapo|dapo-math|dapo_math)
    PRETRAIN_DATASET="dapo"
    PRETRAIN_DATASET_PATH="${PRETRAIN_DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}"
    ;;
  *)
    [[ -n "$PRETRAIN_DATASET_PATH" ]] || { echo "ERROR: unknown PRETRAIN_DATASET; set PRETRAIN_DATASET_PATH" >&2; exit 2; }
    ;;
esac

MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
timestamp="$(date -u +%Y%m%dT%H%M%S)"
short_uuid="$($PYTHON_BIN -c 'import uuid; print(uuid.uuid4().hex[:8])')"
RUN_NAME="${RUN_NAME:-${MODEL_KEY}__${PRETRAIN_DATASET}__eagle3-pretrain__seed${TRAIN_SUBSET_SEED}__${timestamp}__${short_uuid}}"
RUN_DIR="${RUN_DIR:-$MODEL_OUTPUT_ROOT/$RUN_NAME}"

if [[ "${RESUME:-}" == "auto" && -e "$MODEL_OUTPUT_ROOT/active_run" ]]; then
  active="$(readlink -f "$MODEL_OUTPUT_ROOT/active_run")"
  if [[ ! -f "$active/checkpoints/pretrain_complete.json" ]]; then
    RUN_DIR="$active"
    RUN_NAME="$(basename "$RUN_DIR")"
  fi
elif [[ -n "${RESUME:-}" && "${RESUME:-}" != "auto" ]]; then
  RUN_DIR="$(readlink -f "$RESUME")"
  RUN_NAME="$(basename "$RUN_DIR")"
fi

SOURCE_SHAREGPT="$PRETRAIN_DATASET_PATH"
if [[ "$PRETRAIN_DATASET_PATH" == *.parquet || "$PRETRAIN_DATASET_PATH" == *.jsonl ]]; then
  SOURCE_SHAREGPT="$RUN_DIR/data/source_sharegpt.json"
fi
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$([[ "$MODEL_TYPE" == llama ]] && echo llama3 || echo qwen)}"

printf 'Run name : %s\nRun dir  : %s\nModel    : %s\nDataset  : %s\nEpochs   : %s\nGPUs     : %s\n' \
  "$RUN_NAME" "$RUN_DIR" "$MODEL" "$PRETRAIN_DATASET_PATH" "$PRETRAIN_EPOCHS" "$NPROC_PER_NODE"
printf 'Command  : TARGET_MODEL_PATH=%q SHAREGPT_PATH=%q PRETRAIN_ROOT=%q bash %q\n' \
  "$MODEL" "$SOURCE_SHAREGPT" "$RUN_DIR" "$PROJECT_DIR/pretrain_eagle3_sharegpt_b200.sh"
printf 'Pretrain : attention=%s distributed=%s buckets=%s workers=%s effective_batch=%s\n' \
  "${PRETRAIN_ATTENTION_BACKEND:-saved-backend-on-resume-or-fa}" \
  "$PRETRAIN_DISTRIBUTED_MODE" "$PRETRAIN_LENGTH_BUCKETING" "$PRETRAIN_DATALOADER_WORKERS" \
  "$((PRETRAIN_BATCH_SIZE * PRETRAIN_ACCUMULATION_STEPS * NPROC_PER_NODE))"
if [[ "${DRY_RUN:-false}" == "true" ]]; then return 0 2>/dev/null || exit 0; fi

"$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --python-only

[[ -f "$MODEL/config.json" ]] || { echo "ERROR: model config not found: $MODEL/config.json" >&2; exit 2; }
[[ -f "$PRETRAIN_DATASET_PATH" ]] || { echo "ERROR: pretrain dataset not found: $PRETRAIN_DATASET_PATH" >&2; exit 2; }
mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/data"
if [[ "$SOURCE_SHAREGPT" != "$PRETRAIN_DATASET_PATH" && ! -f "$SOURCE_SHAREGPT" ]]; then
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/prepare_local_pretrain_data.py" \
    --input "$PRETRAIN_DATASET_PATH" --output "$SOURCE_SHAREGPT" --max-samples "$PRETRAIN_MAX_SAMPLES"
fi
"$PYTHON_BIN" "$PROJECT_DIR/scripts/write_run_metadata.py" --run-dir "$RUN_DIR" --kind pretrain \
  --item "run_name=$RUN_NAME" --item "model=$MODEL" --item "dataset=$PRETRAIN_DATASET_PATH" \
  --item "learning_rate=$PRETRAIN_LR" --item "batch_size=$PRETRAIN_BATCH_SIZE" \
  --item "accumulation_steps=$PRETRAIN_ACCUMULATION_STEPS" \
  --item "max_length=$PRETRAIN_MAX_LENGTH" --item "epochs=$PRETRAIN_EPOCHS" \
  --item "chat_template=$CHAT_TEMPLATE" \
  --item "nproc_per_node=$NPROC_PER_NODE"

TARGET_MODEL_PATH="$MODEL" SHAREGPT_PATH="$SOURCE_SHAREGPT" MODEL_OUTPUT_ROOT="$MODEL_OUTPUT_ROOT" \
PRETRAIN_ROOT="$RUN_DIR" RUN_ID="$RUN_NAME" RESUME_PRETRAIN=true \
PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS" PRETRAIN_BATCH_SIZE="$PRETRAIN_BATCH_SIZE" \
PRETRAIN_GRADIENT_ACCUMULATION="$PRETRAIN_ACCUMULATION_STEPS" PRETRAIN_LR="$PRETRAIN_LR" \
PRETRAIN_SAVE_INTERVAL="$PRETRAIN_SAVE_INTERVAL" PRETRAIN_SEED="$TRAIN_SUBSET_SEED" \
PRETRAIN_NPROC_PER_NODE="$NPROC_PER_NODE" CAPTURE_NPROC_PER_NODE="$NPROC_PER_NODE" \
PRETRAIN_MAX_LENGTH="$PRETRAIN_MAX_LENGTH" \
PRETRAIN_ATTENTION_BACKEND="$PRETRAIN_ATTENTION_BACKEND" \
PRETRAIN_DISTRIBUTED_MODE="$PRETRAIN_DISTRIBUTED_MODE" \
PRETRAIN_LENGTH_BUCKETING="$PRETRAIN_LENGTH_BUCKETING" \
PRETRAIN_LENGTH_BUCKET_BOUNDARIES="$PRETRAIN_LENGTH_BUCKET_BOUNDARIES" \
PRETRAIN_DATALOADER_WORKERS="$PRETRAIN_DATALOADER_WORKERS" \
PRETRAIN_COMPACT_TEACHER="$PRETRAIN_COMPACT_TEACHER" \
PRETRAIN_OPTIMIZER_CPU_OFFLOAD="$PRETRAIN_OPTIMIZER_CPU_OFFLOAD" \
CAPTURE_MAX_LENGTH="$PRETRAIN_MAX_LENGTH" CHAT_TEMPLATE="$CHAT_TEMPLATE" \
CAPTURE_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" TRAIN_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHON_BIN="$PYTHON_BIN" bash "$PROJECT_DIR/pretrain_eagle3_sharegpt_b200.sh"

"$PYTHON_BIN" "$PROJECT_DIR/scripts/write_run_metadata.py" --run-dir "$RUN_DIR" --kind pretrain --status complete \
  --item "run_name=$RUN_NAME" --item "model=$MODEL" --item "dataset=$PRETRAIN_DATASET_PATH" \
  --item "learning_rate=$PRETRAIN_LR" --item "batch_size=$PRETRAIN_BATCH_SIZE" \
  --item "accumulation_steps=$PRETRAIN_ACCUMULATION_STEPS" \
  --item "max_length=$PRETRAIN_MAX_LENGTH" --item "epochs=$PRETRAIN_EPOCHS" \
  --item "chat_template=$CHAT_TEMPLATE" --item "nproc_per_node=$NPROC_PER_NODE" \
  --item "draft_checkpoint=$RUN_DIR/checkpoints/$RUN_NAME-latest" \
  --item "vocab_mapping=$RUN_DIR/features/vocab_mapping/vocab_mapping.pt"
