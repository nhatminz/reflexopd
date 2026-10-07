#!/usr/bin/env bash
# Quick REAL frozen-rollout sweep. No policy/draft training or downloads.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
export METHOD=opd_reflex
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATASET="${DATASET:-simplelr}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export RESPONSES_PER_PROMPT="${RESPONSES_PER_PROMPT:-8}"
export BENCH_MAX_LENGTH="${BENCH_MAX_LENGTH:-512}"
export BENCH_MAX_PROMPT_LENGTH="${BENCH_MAX_PROMPT_LENGTH:-256}"
export OPD_FAST_LRS="${OPD_FAST_LRS:-0.001,0.01,0.05,0.1}"
export OPD_STREAMS="${OPD_STREAMS:-0,1}"
export BENCH_SEEDS="${BENCH_SEEDS:-42,43}"
export BENCH_ITERATIONS="${BENCH_ITERATIONS:-2}"
export BENCH_WARMUP="${BENCH_WARMUP:-1}"
source "$PROJECT_DIR/configs/_shared/b200_common.env"
source "$PROJECT_DIR/configs/$MODEL_KEY/b200.env"
PYTHON_BIN="${PYTHON_BIN:-python3}"
case "${DATASET,,}" in
 simplelr|simplelr_abel_level3to5) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/simplelr_abel_level3to5/train.parquet}" ;;
 dapo) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet}" ;;
 gsm8k) DATASET_PATH="${DATASET_PATH:-$DATA_ROOT/gsm8k/main/train-00000-of-00001.parquet}" ;;
 *) : "${DATASET_PATH:?set existing DATASET_PATH}" ;;
esac
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs}"
PRETRAIN_MODEL_ROOT="${PRETRAIN_MODEL_ROOT:-$OUTPUT_ROOT/pretrain/$MODEL_KEY}"
export DRAFT_CHECKPOINT="${DRAFT_CHECKPOINT:-$PRETRAIN_MODEL_ROOT/latest_checkpoint}"
export DRAFT_CONFIG="${DRAFT_CONFIG:-$PRETRAIN_MODEL_ROOT/latest_draft_config.json}"
export VOCAB_MAPPING="${VOCAB_MAPPING:-$PRETRAIN_MODEL_ROOT/latest_vocab_mapping.pt}"
export OPD_PROPOSAL_PROFILE_DIR="${OPD_PROPOSAL_PROFILE_DIR:-$OUTPUT_ROOT/benchmarks/opd_proposals}"
BENCH_OUTPUT="${BENCH_OUTPUT:-$OUTPUT_ROOT/benchmarks/opd_${MODEL_KEY}_$(date -u +%Y%m%dT%H%M%S_%N)}"
cmd=("$PYTHON_BIN" "$PROJECT_DIR/scripts/benchmark_opd_reflex.py"
 --target-model "$MODEL" --target-adapter "$TARGET_ADAPTER"
 --draft-checkpoint "$DRAFT_CHECKPOINT" --draft-config "$DRAFT_CONFIG" --vocab-mapping "$VOCAB_MAPPING"
 --dataset-path "$DATASET_PATH" --output "$BENCH_OUTPUT"
 --batch-size "$BATCH_SIZE" --responses "$RESPONSES_PER_PROMPT"
 --max-length "$BENCH_MAX_LENGTH" --max-prompt-length "$BENCH_MAX_PROMPT_LENGTH"
 --temperature "$TEMPERATURE" --top-p "$TOP_P" --attn-implementation "$ATTENTION_IMPLEMENTATION" --dtype "$MODEL_DTYPE"
 --verification-capacity "$VERIFICATION_CAPACITY" --max-verification-num "$MAX_VERIFICATION_NUM"
 --max-draft-k "$MAX_DRAFT_K" --max-draft-length "$MAX_DRAFT_TOKEN_LENGTH"
 --min-draft-length "$MIN_DRAFT_TOKEN_LENGTH" --draft-length-c "$DRAFT_TOKEN_LENGTH_C"
 --rank "$OPD_RANK" --topk "$OPD_TOPK" --fast-lrs "$OPD_FAST_LRS" --streams "$OPD_STREAMS"
 --visited-weight "$OPD_VISITED_WEIGHT" --frontier-weight "$OPD_FRONTIER_WEIGHT"
 --seeds "$BENCH_SEEDS" --iterations "$BENCH_ITERATIONS" --warmup "$BENCH_WARMUP")
if [[ "$OPD_PROFILE" == 1 ]];then cmd+=(--profile);fi
if [[ "$OPD_DIAGNOSTICS" == 1 ]];then cmd+=(--diagnostics);fi
cmd+=("$@")
printf 'Benchmark output: %s\nCommand:' "$BENCH_OUTPUT";printf ' %q' "${cmd[@]}";printf '\n'
if [[ "${DRY_RUN:-false}" == true ]];then "${cmd[@]}" --dry-run;exit 0;fi
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/third_party/SpecForge${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PYTHON_BIN" "$PROJECT_DIR/scripts/validate_environment.py" --require-cuda
exec "${cmd[@]}"
