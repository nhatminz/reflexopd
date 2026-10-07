#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SPECFORGE_DIR="${SPECFORGE_DIR:-$REPO_DIR/third_party/SpecForge}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${PRETRAIN_NPROC_PER_NODE:-${NPROC_PER_NODE:-1}}"
case "$NPROC_PER_NODE" in 1|2|4|8) ;; *) echo "NPROC_PER_NODE must be 1, 2, 4 or 8" >&2; exit 2 ;; esac

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to the production target}"
: "${PRETRAIN_ROOT:?Set PRETRAIN_ROOT to an existing captured/pretrain run}"
DRAFT_CONFIG="${DRAFT_CONFIG:-$PRETRAIN_ROOT/config/eagle3.json}"
FEATURE_DIR="${FEATURE_DIR:-$PRETRAIN_ROOT/features}"
VOCAB_MAPPING="${VOCAB_MAPPING:-$FEATURE_DIR/vocab_mapping/vocab_mapping.pt}"
RUN_ID="benchmark_$(date -u +%Y%m%dT%H%M%S_%N)"
BENCHMARK_OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-$(dirname "$PRETRAIN_ROOT")/$RUN_ID}"
[[ ! -e "$BENCHMARK_OUTPUT_DIR" ]] || { echo "Use a new BENCHMARK_OUTPUT_DIR: $BENCHMARK_OUTPUT_DIR" >&2; exit 2; }
[[ -f "$DRAFT_CONFIG" && -f "$VOCAB_MAPPING" && -d "$FEATURE_DIR" ]] || { echo "Existing config/features/vocab mapping required; benchmark never runs capture." >&2; exit 2; }

export PYTHONPATH="$SPECFORGE_DIR:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PYTHON_BIN" "$REPO_DIR/scripts/validate_environment.py" --require-cuda
mkdir -p "$BENCHMARK_OUTPUT_DIR"
cmd=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone "--nproc_per_node=$NPROC_PER_NODE"
  "$SCRIPT_DIR/benchmark_pretrain.py"
  --config "$SPECFORGE_DIR/examples/configs/offline/colocated/qwen2.5-7b-eagle3-offline.yaml"
  --output-dir "$BENCHMARK_OUTPUT_DIR"
  --steps "${BENCHMARK_STEPS:-30}" --warmup-steps "${BENCHMARK_WARMUP_STEPS:-5}"
  "model.target_model_path=$TARGET_MODEL_PATH"
  "model.draft_model_config=$DRAFT_CONFIG"
  "model.vocab_mapping_path=$VOCAB_MAPPING"
  "data.hidden_states_path=$FEATURE_DIR"
  "data.max_length=2048"
  "data.chat_template=${CHAT_TEMPLATE:-qwen}"
  "data.length_bucketing=${PRETRAIN_LENGTH_BUCKETING:-true}"
  "data.length_bucket_boundaries=${PRETRAIN_LENGTH_BUCKET_BOUNDARIES:-[512,768,1024,1280,1536,1792,2048]}"
  "data.dataloader_num_workers=${PRETRAIN_DATALOADER_WORKERS:-8}"
  "training.num_epochs=${PRETRAIN_EPOCHS:-1}"
  "training.batch_size=${PRETRAIN_BATCH_SIZE:-4}"
  "training.accumulation_steps=${PRETRAIN_GRADIENT_ACCUMULATION:-${PRETRAIN_ACCUMULATION_STEPS:-1}}"
  "training.learning_rate=${PRETRAIN_LR:-5e-5}"
  "training.seed=${PRETRAIN_SEED:-${TRAIN_SUBSET_SEED:-42}}"
  "training.ttt_length=7"
  "training.attention_backend=${PRETRAIN_ATTENTION_BACKEND:-fa}"
  "training.distributed_mode=${PRETRAIN_DISTRIBUTED_MODE:-auto}"
  "training.compact_teacher=${PRETRAIN_COMPACT_TEACHER:-false}"
  "training.optimizer_cpu_offload=${PRETRAIN_OPTIMIZER_CPU_OFFLOAD:-false}"
  "deployment.trainer.nproc_per_node=$NPROC_PER_NODE"
  "run_id=$RUN_ID" "output_dir=$BENCHMARK_OUTPUT_DIR"
)
# Optional initial weights/extra typed overrides retain the same warm-start
# path as production. Output/resume isolation is validated by the runner.
if [[ -n "${INITIAL_DRAFT_CHECKPOINT:-}" ]]; then
  cmd+=("model.draft_checkpoint_path=$INITIAL_DRAFT_CHECKPOINT")
fi
cmd+=("$@")
printf 'Run:'; printf ' %q' "${cmd[@]}"; printf '\n'
"${cmd[@]}" 2>&1 | tee "$BENCHMARK_OUTPUT_DIR/console.log"
