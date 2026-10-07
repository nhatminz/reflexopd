# Chạy FastGRPO và OPD Reflex trên B200

Dùng venv SpecNaacl đã được kiểm tra; KHÔNG dùng venv TLT (khác Torch/Transformers).
Paths model/data/pretrained draft giữ nguyên. Không cần pretrain lại. Pipeline
không tải model/dataset từ Internet; giữ nguyên dependencies trong ENVIRONMENT.md.

Current memory revision và benchmark trước/sau:
[OPD_MEMORY_LAST_THREE.md](OPD_MEMORY_LAST_THREE.md).

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export ATTENTION_IMPLEMENTATION=sdpa # target; eager vẫn override được
export DATASET=simplelr
export TARGET_LR=1e-6
export DRAFT_LR=1e-4
export DRAFT_ACCUMULATION_STEPS=1
export OPD_FAST_LR=0.01   # cũng hỗ trợ FAST_LR nếu OPD_FAST_LR chưa được set
export BATCH_SIZE=8
export ACCUMULATION_STEPS=4
export RESPONSES_PER_PROMPT=8
export OPD_RANK=8
export OPD_TOPK=16
export OPD_VISITED_WEIGHT=1.0
export OPD_FRONTIER_WEIGHT=1.0
export OPD_UPDATE_STREAM=1   # B200/Qwen2.5-3B observed fastest; vẫn override được
export OPD_TRAIN_PROJECTOR=1 # A học tại draft optimizer boundary, không reset
export OPD_PROJECTOR_LR="" # optional A LR; empty inherits draft LR, separate from B_fast LR
export ROLLOUT_LOG_FLUSH_INTERVAL=1 # buffer CSV; tăng lên 8 nếu shared storage chậm
export OPD_PROPOSAL_MODE=auto
export OPD_DENSE_IMPLEMENTATION=auto
export OPD_PROFILE=0
export OPD_DIAGNOSTICS=0
export OPD_KV_MAX_RETAINED_TOKENS=0 # keep high-water KV pools; optional retention cap
```

Dataset default:
`/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet`.
Đổi bằng `DATASET=dapo`/`gsm8k`, hoặc export `DATASET_PATH` tới file thật.
Draft/config/mapping default: `outputs/pretrain/<model_key>/latest_*`.
Nếu khác, export DRAFT_CHECKPOINT, DRAFT_CONFIG, VOCAB_MAPPING của cùng pretrained run.

## Chạy từng cặp model

```bash
# OPD Reflex                       # FastGRPO/OFF
bash train_qwen25_3b.sh             # bash train_qwen25_3b_fastgrpo.sh
bash train_qwen3_1p7b.sh             # bash train_qwen3_1p7b_fastgrpo.sh
bash train_qwen3_4b.sh               # bash train_qwen3_4b_fastgrpo.sh
bash train_qwen25_1p5b.sh            # bash train_qwen25_1p5b_fastgrpo.sh
bash train_qwen25_7b.sh              # bash train_qwen25_7b_fastgrpo.sh
bash train_qwen25_14b.sh             # bash train_qwen25_14b_fastgrpo.sh
bash train_llama31_8b.sh             # bash train_llama31_8b_fastgrpo.sh
```

Mỗi lệnh là một run riêng. Run name tự có method/seed/timestamp/UUID.
Không gọi tất cả model nếu chỉ cần một model.

Dry-run và smoke thực với weights thật:

```bash
DRY_RUN=true bash train_qwen25_3b.sh
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b.sh --max_grpo_steps 2
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b_fastgrpo.sh --max_grpo_steps 2
```

## Sweep nhanh LR/stream: FROZEN model, không train/checkpoint

Quy trình autotuning mới đầy đủ: [OPD_AUTOTUNING.md](OPD_AUTOTUNING.md).
Tune6 models trong một lần, dedup theo execution key. Profile cũ bị reject;
launcher/sweep tự tìm đúng profile, không cần export một profile theo model name.

```bash
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT
bash scripts/tune_opd_proposals.sh
export MODEL_KEY=qwen3_1p7b # hoặc qwen25_3b / qwen3_4b

# KV memory/latency, không model forward hay training. Đọc config target/draft.
KV_BENCH_LENGTHS=256,512,1024,2048 KV_BENCH_BATCHES=16,32,64 \
bash scripts/benchmark_opd_kv.sh
```

KV output: `outputs/benchmarks/kv_<model_key>_<timestamp>/{target,draft}/report.json`
và `kv.csv`. Full-layer memory mặc định; OOM được ghi trong report. Muốn kiểm tra
component nhỏ trước: `KV_BENCH_LAYERS=1 bash scripts/benchmark_opd_kv.sh` (không
được xem như memory của full model). Cache không realloc khi row finish; geometric
growth/row remap vẫn copy live prefix, telemetry không che giấu các copies này.
Model khác có cùng execution key được reuse; không reuse sai GPU/V/rank/dtype/kernel.

```bash
OPD_FAST_LRS=0.001,0.01,0.05,0.1 OPD_STREAMS=0,1 \
BENCH_SEEDS=42,43 BENCH_ITERATIONS=2 BENCH_WARMUP=1 \
bash scripts/sweep_opd_reflex.sh
```

Quick defaults: prompts8, responses8, total max_length512 (bao gồm prompt),
max_prompt_length256, verification_capacity512, K8, depth5. Warmup chạy đúng
seed/prompt schedule để tránh đổ JIT vào timing. Muốn workload train:

```bash
BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=256 \
BENCH_ITERATIONS=3 BENCH_SEEDS=11,29,47 bash scripts/sweep_opd_reflex.sh
```

Thêm `--gpu-utilization` để benchmark-only lấy nvidia-smi utilization. Để đo riêng
OPD/KV compaction trong diagnostic replay (excluded from wall/throughput):

```bash
OPD_PROFILE=1 OPD_FAST_LRS=0.01 OPD_STREAMS=0,1 BENCH_SEEDS=42 \
BENCH_ITERATIONS=1 BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=256 \
bash scripts/sweep_opd_reflex.sh --gpu-utilization
```

Report/summary có `kv_cache_bytes`, `host_syncs_per_round`, copy/reallocation
counters và `kv_compaction_profile_ms` khi profile được bật. OPD còn1 scheduling
host packet/round; counters baseline không instrument là null. PROFILE=0 khi train.

Prompt đã dài bằng max_length bị reject; tăng max_length hoặc giảm max_prompt_length.
Dùng BENCH_OUTPUT mới nếu muốn chỉ định output. Script xuất report.json,
summary.csv, responses.jsonl và fastest_observed.env. recommendation=null nếu
không config nào đồng thời AAL tăng và throughput>=baseline. fastest_observed.env
KHÔNG được coi là recommendation mặc định; kiểm tra delta/overhead, sau đó validate
trên held-out prompts trước khi tự source nó.

OPD_PROFILE=1 chạy diagnostic replay TÁCH RIÊNG khỏi wall/throughput. Các profile
timers không phải authoritative speedup. OPD_DIAGNOSTICS=1 thêm B norm/max ở cuối
rollout, không thêm target/draft transformer forward.

## Tune sparse/dense trên B200 và benchmark end-to-end

```bash
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT
# Mặc định cả6 model, không bị MODEL_KEY của training trước đó giới hạn.
CUDA_VISIBLE_DEVICES=0 bash scripts/tune_opd_proposals.sh
# Hoặc chỉ một model / custom model:
OPD_TUNE_MODELS=qwen3_1p7b CUDA_VISIBLE_DEVICES=0 bash scripts/tune_opd_proposals.sh
```

Script đọc real config/checkpoint/mapping và detect V/rank/dtype; không load target
hay draft transformer để benchmark. Shared profiles ở `outputs/benchmarks/opd_proposals/`
với GPU/cc/V/r/dtype/TopK/kernel/compiler key, không key theo tên model/H. Summary
`models_summary.json` ghi profile dùng cho từng model. Missing/corrupt resources
warning+skip; profile compatible đã tồn tại không tune lại.
Valid profile interpolates measured costs theo contexts và active rows để chọn
sparse/fused/GEMM. Default không tự tune dài trong train; có thể opt-in
`OPD_AUTO_TUNE_IF_MISSING=1`. Explicit profile sai key bị reject; missing exact
profile warning+uncalibrated safe fallback. Không deploy RTX3090 profile lên B200.

Tiếp theo:

```bash
export OPD_PROPOSAL_MODE=auto
export OPD_DENSE_IMPLEMENTATION=auto
MODEL_KEY=qwen3_1p7b OPD_FAST_LRS=0.001,0.01,0.05 OPD_STREAMS=0,1 \
BENCH_SEEDS=42,43 BENCH_ITERATIONS=2 BENCH_MAX_LENGTH=2048 \
bash scripts/sweep_opd_reflex.sh --gpu-utilization

# Sau khi xem report.json/summary.csv; giữ cả delta AAL âm nếu có.
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b.sh
# Baseline historical, cùng model/data/sampling/training settings:
# Hiện là shared optimized FastGRPO, giữ historical proposal/RNG semantics.
CUDA_VISIBLE_DEVICES=0 bash train_qwen3_1p7b_fastgrpo.sh
```

Context/active-row trial grids sinh geometric từ batch*responses*max_draft_k và
actual V; không hard-code absolute context/slot sizes. OPD_TUNE_SHAPES/SLOTS là
explicit override nếu bạn cần workload riêng. Tuner đo correction/Top16/normalization,
không đo feature projection chung, do đó hidden size khác được reuse cùng key.
--gpu-utilization lấy mẫu nvidia-smi mỗi 0.5s, có thể ảnh hưởng nhẹ CPU timing;
bỏ flag để đo throughput riêng. OPD_PROFILE=1 thêm replay profiling tách khỏi
wall/throughput chính. Sweep ghi outputs/benchmarks/opd_<model>_<timestamp>/.
Một sweep frozen không train A: dùng checkpoint OPD đã train khi đánh giá learned A.

FastGRPO từ nay là historical native TopK(draft_k), không dùng shared Top16.
OPD luôn fused Top16 kể cả cold; tied candidate IDs khác historical được chấp
nhận. Không thêm fallback chậm. A phải qua optimizer steps mới gọi là đã train;
muốn đánh giá learned A hãy dùng DRAFT_CHECKPOINT của run OPD đã train. Sweep
frozen không tự train A/policy/draft. Một scheduling host boundary vẫn còn để
giữ đúng dynamic batch/HF crop/RNG; không claim toàn decoder zero-sync.
UPDATE_STREAM=0: packet count sau update chính xác cho proposal kế tiếp.
UPDATE_STREAM=1: giữ overlap, dùng snapshot một round cũ + safe scratch bound;
không thêm wait để lấy current count chỉ cho dispatch. Chi tiết trong OPD_AUTOTUNING.md.
Rollout CSV thêm iter_opd_fused_rounds/gemm_rounds; dense_rounds giữ backward compatibility.

## Outputs và plot AAL đúng từng step

Train outputs: `outputs/train/<model_key>/<run_name>/`.
Mỗi completed inherited GRPO step có một row trong logs/timing.csv và một
target_train row trong logs/metrics.jsonl (các phase khác có row riêng).
Checkpoint target/draft/resume nằm trong checkpoints/ của chính run.
Benchmark outputs: `outputs/benchmarks/opd_<model_key>_<timestamp>/`.

Sau khi có hai run, truyền đường dẫn logs thật của chúng:

```bash
python scripts/plot_opd_aal.py --fastgrpo "$FASTGRPO_TIMING_CSV" \
  --opd-reflex "$OPD_TIMING_CSV" --output "$NEW_AAL_PLOT_PATH"
```

Các biến trên phải trỏ tới CSV/output bạn thực sự chọn. Plot dùng step counters,
không dùng cumulative AAL/moving average. Các pretrain launchers vẫn giữ nguyên.
Không resume optimizer trajectory của method cũ vào OPD; giữ cùng method khi
resume. Có thể dùng lại pretrained draft/target adapter như initialization.
Checkpoint trước learned-A không dùng làm optimizer resume; dùng draft weights
của nó để bắt đầu run mới. Checkpoint mới lưu pending A gradient riêng từng rank.

## Fix `confidence-selected draft tree is not parent-closed`

Đây là lỗi tie trong pruning tree OPD, không phải thiếu FlashAttention hay PEFT.
Đồng bộ toàn bộ `helper/` của bản sửa (cần đủ tree_verification.py,
tree_kernels.py, opd_reflex.py và specualtive_generate.py), cùng
`tests/opd_fixtures.py` và `tests/test_tree_confidence_selection.py`.
Không chỉ copy một file; không tắt assertion để vượt lỗi.

Sau khi process bị device assert đã thoát, dùng process Python mới:

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
export PYTHON_BIN="$(command -v python)"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -m pytest -q \
  tests/test_tree_confidence_selection.py tests/test_opd_proposal_lifetime.py

# Hai GRPO steps thực trước khi chạy dài, model/data/draft giữ nguyên.
CUDA_VISIBLE_DEVICES=0 DATASET=gsm8k MAX_TRAIN_SAMPLES=128 \
  PYTHON_BIN="$PYTHON_BIN" bash train_qwen25_1p5b.sh --max_grpo_steps 2

# Full run chỉ sau khi smoke thành công.
CUDA_VISIBLE_DEVICES=0 DATASET=gsm8k PYTHON_BIN="$PYTHON_BIN" \
  bash train_qwen25_1p5b.sh
```

Pruning mới giữ confidence FP32 nguyên vẹn, dùng node index làm secondary key
để cha thắng con khi tie. Không đổi target sampling hoặc historical FastGRPO.

Bản sửa tiếp theo còn giữ riêng root confidence trong buffer `B*K` preallocated:
`OPDReflex.propose()` trả view tạm thời, bị ghi đè ở proposal kế tiếp. Không giữ
view này trong history confidence; không clone toàn vocab. Cần đồng bộ thêm
`helper/opd_reflex_kernels.py` và `tests/test_opd_proposal_lifetime.py` cùng các
file trên. Test mới thay đổi distribution theo depth và batch row để bắt lỗi
alias mà test distribution cố định không phát hiện. Kernel cũng xử lý đúng
tile toàn `-inf` (mass=0) và không emit lại token đã chọn ở các slot probability0.
Nếu đang dùng profile adaptive cũ, retune sau khi cập nhật kernel hash; không
tắt fingerprint check để dùng profile từ revision khác.

## Target attention: SDPA default

Theo yêu cầu mới, shared config và direct training/benchmark CLI đều mặc định
`sdpa` thay vì `eager`. Draft EAGLE-3 runtime/online trainer vốn đã dùng SDPA.
Không cần cài thêm flash-attn để gọi PyTorch SDPA. Custom tree mask vẫn giữ
nguyên; PyTorch chọn kernel thích hợp, không đảm bảo luôn là flash kernel.
Backend này có thể thay đổi floating-point rounding/tokens so với eager;
không coi hai backend là bitwise-identical hoặc exact resume qua backend khác.
Đo cùng backend cho OPD/FastGRPO, không ghép eager baseline với SDPA OPD.

```bash
CUDA_VISIBLE_DEVICES=0 ATTENTION_IMPLEMENTATION=sdpa MAX_TRAIN_SAMPLES=128 \
  PYTHON_BIN="$PYTHON_BIN" bash train_qwen3_1p7b.sh --max_grpo_steps 2
CUDA_VISIBLE_DEVICES=0 ATTENTION_IMPLEMENTATION=sdpa \
  PYTHON_BIN="$PYTHON_BIN" bash train_qwen3_1p7b.sh

# Khi muốn giữ backend trước đây:
ATTENTION_IMPLEMENTATION=eager bash train_qwen3_1p7b.sh
```

Log startup phải ghi `attn_impl=sdpa`; benchmark sweep dùng cùng biến
`ATTENTION_IMPLEMENTATION`. Native Qwen/B200 SDPA training chưa được đo tại
máy phát triển vì weights/data production không có ở đây.
