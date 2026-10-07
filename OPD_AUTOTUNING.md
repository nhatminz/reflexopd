# OPD proposal autotuning (execution-keyed, multi-model)

Memory lifecycle/metadata revision và default stream1:
[OPD_MEMORY_LAST_THREE.md](OPD_MEMORY_LAST_THREE.md). Autotuner/dispatcher algorithm
không đổi trong memory revision; conservative kernel fingerprint cần regenerate.

Production default: `OPD_PROPOSAL_MODE=auto`, `OPD_DENSE_IMPLEMENTATION=auto`.
Historical FastGRPO, OPD objective, B update, target sampling/RNG, verifier and
online draft training remain unchanged. No extra model forward is used for tuning
or runtime dispatch. Microbenchmark consumes already-projected U; feature
projection is common to all backends and not part of crossover measurement.

## Quy trình B200

Đồng bộ toàn bộ code/config/scripts sửa trong revision này, không chỉ một `.sh`.
Dùng venv SpecNaacl đang dùng, không thay bằng venv TLT. Các paths model/data và
pretrained draft giữ nguyên. Không cần pretrain lại.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export MODEL_DTYPE=bf16
export OPD_RANK=8
export OPD_TOPK=16
export BATCH_SIZE=8
export RESPONSES_PER_PROMPT=8

# Profile format/hash cũ không dùng được. Không export profile của một model
# sang tất cả model; để discovery tự tìm theo execution key.
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT

# Một lệnh kiểm tra 6 models; mỗi UNIQUE config chỉ benchmark một lần.
bash scripts/tune_opd_proposals.sh
```

Default models: qwen25_1p5b/3b/7b/14b, qwen3_1p7b/4b. `MODEL_KEY` đã export cho
training KHÔNG làm tuner mặc định chỉ chạy một model. Missing/corrupt checkpoint,
config/mapping: warning + skip; các model hợp lệ vẫn tiếp tục. Nếu không có model
hợp lệ, không tạo measured profile giả. Progress: Models inspected, Unique configs,
Contexts, Active rows. Có resources rồi chạy lại: exact profile đã có được reuse.

Tuner đọc `outputs/pretrain/<key>/latest_{checkpoint,draft_config.json,vocab_mapping.pt}`.
Checkpoint shape/dtype đọc bằng meta/mmap hoặc safetensors headers, không load model
lên GPU; mapping d2t/t2d được validate. Saved projector rank được detect; OPD_RANK
explicit khác learned projector bị skip/error, không bỏ learned A để ép rank khác.
Execution dtype dùng MODEL_DTYPE (bf16 default như training), không nhầm với dtype
của weights trên disk; checkpoint_dtype được ghi trong summary.

Profile output: `outputs/benchmarks/opd_proposals/<GPU>__cc...__V...__r...__dtype__k...__hash...json`.
`models_summary.json` ghi model -> V/r/dtype -> profile used; JSON profile ghi measured
times/threshold summaries, dense implementation, execution metadata, models inspected.
Hai model cùng key trỏ tới đúng một profile; hidden/model size không là key vì proposal
chỉ nhận logits[V] và U[rank]. Key còn gồm TopK, Torch/Triton/CUDA versions để tránh
reuse sai compilation/runtime. Kernel hash bao gồm correction và merge kernels.

## Benchmark end-to-end rồi smoke/train

```bash
export OPD_PROPOSAL_MODE=auto
export OPD_DENSE_IMPLEMENTATION=auto
export OPD_PROFILE=0
export OPD_DIAGNOSTICS=0
export OPD_AUTO_TUNE_IF_MISSING=0 # không tự benchmark dài trong training job

MODEL_KEY=qwen3_1p7b OPD_FAST_LRS=0.001,0.01,0.05 OPD_STREAMS=0,1 \
BENCH_SEEDS=42,43 BENCH_ITERATIONS=2 BENCH_MAX_LENGTH=2048 \
BENCH_MAX_PROMPT_LENGTH=256 bash scripts/sweep_opd_reflex.sh --gpu-utilization

# Smoke: dùng LR/stream bạn chọn sau khi xem report (đây vẫn là default chưa tune).
TARGET_LR=1e-5 DRAFT_LR=1e-5 OPD_FAST_LR=0.01 OPD_UPDATE_STREAM=1 \
MAX_TRAIN_SAMPLES=128 bash train_qwen3_1p7b.sh --max_grpo_steps 2

# Full training, chỉ tự launch sau khi smoke thành công.
export OPD_UPDATE_STREAM=1
bash train_qwen3_1p7b.sh
# Historical baseline, run riêng cùng settings:
bash train_qwen3_1p7b_fastgrpo.sh
```

Đổi MODEL_KEY/launcher sang qwen25_3b, qwen3_4b hoặc các cặp model khác tương ứng.
Không export DRAFT_CONFIG/CHECKPOINT/MAPPING của model trước khi chuyển sang model
khác, trừ khi bạn chủ động override về đúng pretrained resources của model mới.
Launcher tự lookup exact key trước load model; runtime direct Python/sweep cũng
discover tại setup. Log mode/profile/GPU/cc/V/rank/dtype/hash. Explicit profile sai
bị reject; automatic lookup không tìm được thì warning rõ và dùng uncalibrated
safe fallback, không giả vờ tối ưu B200. `OPD_AUTO_TUNE_IF_MISSING=1` là opt-in chạy
tuner tại launcher startup trước load weights, không trong generation hot path.

Outputs training: `outputs/train/<key>/<run>/logs/{metrics.jsonl,timing.csv,rollout_timing.csv}`.
Sweep: `outputs/benchmarks/opd_<key>_<timestamp>/{report.json,summary.csv,responses.jsonl}`.
Rollout CSV giữ `iter_opd_sparse_rounds`, `iter_opd_fused_rounds`, `iter_opd_gemm_rounds`,
`iter_opd_active_rows_mean/max`, và legacy `iter_opd_dense_rounds=fused+gemm`.
Fused/GEMM counts dùng backend đã biết phía host; sparse/legacy-dense giữ counters
cũ. Không thêm GPU atomic, kernel hoặc GPU transfer chỉ để log. Counts theo ROOT
proposal/tree build, không cộng mọi branch proposal.
Cold B=0 không chạy correction preparation và được ghi raw/sparse, giữ fast Top16.

## Tune riêng/custom resources

```bash
# Chỉ model này; profile vẫn shared execution-key, không key bằng model name.
OPD_TUNE_MODELS=qwen3_1p7b bash scripts/tune_opd_proposals.sh
# Model bất kỳ, gồm llama hiện có:
OPD_TUNE_MODELS=llama31_8b bash scripts/tune_opd_proposals.sh
# Inspect không benchmark/không ghi profile:
bash scripts/tune_opd_proposals.sh --inspect-only
# Re-measure có chủ đích, overwrite đúng profile config được chọn:
OPD_TUNE_MODELS=qwen3_1p7b bash scripts/tune_opd_proposals.sh --force
```

Overrides: OUTPUT_ROOT/PRETRAIN_ROOT, OPD_PROPOSAL_PROFILE_DIR, OPD_TUNE_MODELS,
OPD_RANK, MODEL_DTYPE/OPD_TUNE_DTYPE, OPD_TOPK, BATCH_SIZE, RESPONSES_PER_PROMPT,
MAX_DRAFT_K, OPD_TUNE_CONTEXT_POINTS/ACTIVE_POINTS/ITERATIONS. Contexts geometric
theo batch*responses*max_draft_k; active trials geometric theo actual V, gồm0/V/TopK.
Không có absolute contexts128/256/512 hay active rows1024/4096 heuristic.
Explicit OPD_TUNE_SHAPES (`2x3,7x5`) / OPD_TUNE_SLOTS (`0,16,V`) chỉ cho workload
bạn chủ động chọn. DRAFT_CONFIG/CHECKPOINT/VOCAB_MAPPING hoặc PRETRAIN_MODEL_ROOT
overrides chỉ áp dụng khi OPD_TUNE_MODELS chỉ có một model. OPD_TUNE_OUTPUT optional
single-profile path, không dùng cho multi-config run; default shared directory.

## Dispatch và giới hạn đồng bộ

Valid profile dùng interpolation của measured costs theo log(context_count) và
linear active_rows; ngoài range clamp nearest measured bucket, không V/8 fallback.
Host selection memoized; chỉ sparse OR fused OR GEMM correction được launch.
Sparse scratch giữ bound an toàn cho async metadata cũ; kernel luôn đọc B/active
set hiện tại, không approximate logits hay đổi objective.

Scheduling vẫn đúng một tiny D2H packet/verification round. Với UPDATE_STREAM=0,
packet đọc current active_count sau update: chính xác cho proposal kế tiếp, không
thêm sync. Với UPDATE_STREAM=1, **giữ overlap**: packet đọc root pre-update snapshot
ổn định (một round cũ). Exact count mới tại packet đó sẽ phải wait update, phá overlap;
revision này không thêm wait để dispatch. Đây là giới hạn của yêu cầu đồng thời
"exact current count" và "không làm chậm async path"; theo lựa chọn của bạn,
ưu tiên throughput/overlap và cho phép strategy lệch một round.
Count cũ chỉ ảnh hưởng strategy/performance; outputs vẫn exact. Wait B chỉ trước
proposal thực sự đọc B như trước. Không có dispatch `.item()`/extra D2H.

## Validation

Tests gồm execution-key mismatch/discovery, ranks/vocabs/dtypes, interpolation,
hidden-independent dedup/reuse, missing/corrupt resources, Torch/safetensors
checkpoint headers, synchronous/async packet count, only-selected launches và
sparse/fused/GEMM parity. CUDA tiny tuner thực và native HF+EAGLE3 rollout smoke
chạy trên RTX3090; B200 production checkpoints/data không có tại máy sửa code.
Không claim B200 throughput/AAL improvement hay "không regression trên mọi GPU"
khi chưa có end-to-end measurement. Xem IMPLEMENTATION_REPORT.md.
