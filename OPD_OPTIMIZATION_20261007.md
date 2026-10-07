# OPD: growable KV, reused masks, compact sampler metadata, iteration logs

Autotuner/dispatcher đã được nâng cấp sau phần memory revision dưới đây.
Current tuning/discovery commands: [OPD_AUTOTUNING.md](OPD_AUTOTUNING.md).
Các ghi chú thresholds/profile layout cũ ở dưới chỉ mô tả revision trước.

Phạm vi: chỉ OPD generation và telemetry chung. Không đổi historical FastGRPO
generation, reward, target sampler/RNG, verifier, GRPO loss hay EAGLE training
schedule. Không đổi dependencies hoặc các đường dẫn model/data/pretrained draft.

## Những thay đổi đã implement

- `helper/opd_static_cache.py`: Cache API của Transformers 5.12.1, growable K/V
  pools cho target và EAGLE draft. Initial capacity 256 token (hoặc prompt thực
  làm tròn theo chunk256 nếu lớn hơn); tăng geometric khi append thực sự vượt
  capacity, chỉ copy prefix đang sống. Không reserve theo max_length*tree_depth.
  Append ghi suffix; rollback/crop đổi length;
  attention chỉ nhìn prefix hợp lệ, không attention vào toàn capacity. EAGLE tree
  expansion rollback về prefix ban đầu, không làm committed cache dài thêm.
  Accepted target suffix vẫn gather vào scratch nhỏ rồi copy in-place.
- `helper/opd_kv_kernels.py`: repeat/select ghi trở lại SAME KV pool, không tạo
  full-capacity pool mới khi row finish. CUDA gather/scatter bằng hai kernel
  stream-ordered, scratch live-prefix growable dùng chung giữa các layers. CPU
  oracle dùng token tiles. Pre-reserve batch axis theo số responses thực để repeat
  không grow pool. Weak owner backlink tránh giữ pool đến cyclic GC cuối rollout.
- `helper/opd_attention{,_kernels}.py`: grow/reuse committed-draft và tree masks,
  expansion mask, token/position buffers, cached position ramp. CUDA causal mask
  ghi một lần vào workspace; không full/repeat large mask mỗi round. Past-position
  buffer có storage riêng, không alias position workspace được ghi lại.
- `helper/opd_sampling.py`: cùng FP32 temperature, softmax, top-p/top-k operations
  và multinomial như sampler cũ. Callback GPU reuse sorted IDs/probabilities đã có,
  trace path một lần, chọn states và extract teacher vào independent buffers
  [selected-state capacity,16] + compact_mass. Full sorted arrays không escape
  sampler. Reuse probability storage khi scatter thay vì zeros_like mới. Full
  vocab/permutation lấy teacher Top16 từ sampler prefix, compact_mass=1; greedy
  lấy trực tiếp sampled ID. Không thêm target/draft transformer forward hay sort.
- Positive ties ở biên teacher được merge bằng integer probability/compact-ID key,
  giữ low-compact-ID tie rule; token p=0 không hợp lệ. Không sort/softmax lại vocab.
  **Ngoại lệ cần nói rõ:** khi positive boundary tie rất lớn, phải đọc phần tied
  suffix để giữ exact tie rule; trường hợp uniform có thể đọc cả sorted support.
  Với compact subset hoặc sampling không tạo sorted intermediate, vẫn scan compact
  probabilities của **selected states**, không phải mọi verification row.
- `helper/opd_reflex{,_kernels}.py`: sparse scores [contexts, capacity-S], active
  slot mapping [V]; không reserve dense [contexts,V] cho sparse/fused-dense. S reserve
  gồm số active đã biết + bound cho một feedback update chưa phản ánh vào host packet;
  tăng capacity theo cấp số nhân, không allocate mỗi round. Dense GEMM workspace
  chỉ tạo khi thực sự được chọn và reuse. Dispatch chỉ launch một backend.
- Teacher/head/union/B update/A terms xử lý GPU-selected work queue bằng persistent
  workers, không launch K*all-tree-rows rồi skip. A reduction chỉ đọc selected rows.
  Union/dedup/p/q/tail/KL/q-p vẫn fused như trước; không thêm heuristic gating.
- `helper/specualtive_generate.py`, `helper/tree_verification.py`: reuse candidate,
  parent/context/token/position/confidence pools, branch TopK output, ping-pong
  ancestry indices, draft attention mask và packed-tree buffers. Bỏ Python history
  lists và các concat ở tree expansion. Native small-tree TopK/gather vẫn có
  internal scratch; không claim zero allocations cho toàn decoder.
- `helper/opd_optimizer.py`: optional separate A LR; giữ AdamW moments khi migrate
  checkpoint single-group sang two-group. Resume two-group khi không set LR mới
  vẫn giữ LR đã lưu. A persistent; B reset mỗi rollout; không optimizer/backward/
  all-reduce mỗi round. Unset LR giữ setup cũ.

## Đồng bộ và memory: giới hạn thực tế

Scheduling host sync/verification round: **1 trước, 1 sau** (tiny D2H packet).
Không thêm scalar GPU read để chọn backend, chọn teacher hoặc ghi CSV. Update stream
vẫn wait ngay trước proposal tiếp theo; cả stream=0/1 có parity tests.
Prefill và natural end-rollout vẫn có transfers. Optional profiling có timing sync
riêng; production PROFILE=DIAGNOSTICS=STATISTICAL_TIME=0 không thêm sync.

KV không copy full prefix **khi append/crop/accept mỗi round trong capacity**.
Geometric grow, prefill repeat và finished-response batch compaction vẫn copy
live prefixes; counters ghi thật các copies này. Không copy phần unused capacity,
không realloc KV pool khi finish. Row compaction vẫn O(live history), KHÔNG claim
latency history-independent: layout attention flat HF cần survivors contiguous.
Scratch CUDA gồm live KV của một layer và dùng chung giữa layers, không full
capacity scratch cho mọi layer; pool batch capacity giữ nguyên sau compaction.
Chỉ full-vocab probability tensor còn sống đến feedback để lookup p của DraftTop16;
không giữ sorted_probs/sorted_ids sau sampler. Subset mapping/unfiltered sampler
vẫn extract teacher từ existing selected-state probabilities, không second sort.
Alias `teacher` cũng được bỏ sau enqueue (record_stream bảo vệ async reader),
không pin probabilities của round cũ đến sampler round sau. Release whole-tree
activations sau accepted-feature gather và temporary KV views trước geometric grow.
B200 peak VRAM với model lớn chưa đo.
Thay layout KV và reduction grad_A giữ cùng toán học, nhưng không hứa bitwise
trajectory trên mọi GPU/attention kernel hoặc qua các checkpoint revision.

## CSV theo DataLoader iteration

`outputs/train/<model_key>/<run_name>/logs/rollout_timing.csv`:
đúng một row cho mỗi iteration được thực thi, kể cả reward std=0, prompt quá dài,
answer None hoặc GRPO step chưa tăng. Resume bỏ qua các iteration đã checkpoint,
không ghi trùng. `global_iter` liên tục qua epochs/resume; resume checkpoint cũ
không có telemetry dùng iterator position và cumulative counters cũ.

- iter_aal = total_acc_length / total_decoded_token_num của riêng rollout.
- cumulative_aal = tổng accepted lengths / tổng sequence verification rounds.
- Accepted-length definition kế thừa FastGRPO, gồm target-sampled root/bonus
  theo committed path; không average các batch AAL ratios.
- Acceptance rate dùng accepted_draft / proposed_draft, khác AAL.
- Wall time từ lúc Python training process bắt đầu đến cuối iteration, gồm
  model/setup, generation, reward, training và checkpoint work đã hoàn thành;
  resume cộng elapsed đã lưu. Downtime giữa các phiên không tính.
- OPD fields dùng host counters ở natural rollout boundary. Không giữ GPU history
  trong telemetry. File handle buffered, flush mỗi ROLLOUT_LOG_FLUSH_INTERVAL,
  sau checkpoint, exception/KeyboardInterrupt, SIGTERM và kết thúc.
- Multi-rank: mỗi rank có file riêng (`rollout_timing.rankN.csv`); file không suffix
  là rank0, không phải all-rank sum. Không thêm all-reduce chỉ để log.
- `timing.csv` và `metrics.jsonl` per-GRPO-step vẫn giữ.

## B200: lệnh chạy

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export MODEL_KEY=qwen3_1p7b

# Dùng config pretrained thật của model; tự đọc compact vocab, hidden size.
# Kernel fingerprint thay đổi ở revision này: KHÔNG reuse profile revision cũ.
export OPD_TUNE_OUTPUT="$PWD/outputs/benchmarks/opd_proposals/${MODEL_KEY}_$(date -u +%Y%m%dT%H%M%S_%N).json"
bash scripts/tune_opd_proposals.sh
export OPD_PROPOSAL_PROFILE="$OPD_TUNE_OUTPUT"

# KV component: target + draft config thật, lengths256/512/1024/2048,
# batches16/32/64. Full layers mặc định; OOM được ghi thật, không thay bằng số giả.
bash scripts/benchmark_opd_kv.sh

# Đo frozen-rollout với historical FastGRPO + OPD, response dài, 2 stream modes.
# Không dùng kết quả timing từ diagnostic replay để claim speedup.
OPD_FAST_LRS=0.001,0.01,0.05 OPD_STREAMS=0,1 BENCH_SEEDS=42,43 \
BENCH_ITERATIONS=2 BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=256 \
bash scripts/sweep_opd_reflex.sh --gpu-utilization

# Tách replay profiling để đo compaction/OPD sections, không lấy replay làm timing win.
OPD_PROFILE=1 OPD_FAST_LRS=0.01 OPD_STREAMS=0,1 BENCH_SEEDS=42 \
BENCH_ITERATIONS=1 BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=256 \
bash scripts/sweep_opd_reflex.sh --gpu-utilization

# Chọn LR/stream sau khi xem report.json; dưới đây vẫn là defaults chưa tune.
OPD_PROJECTOR_LR=1e-5 OPD_FAST_LR=0.01 OPD_UPDATE_STREAM=0 \
ROLLOUT_LOG_FLUSH_INTERVAL=8 bash train_qwen3_1p7b.sh
bash train_qwen3_1p7b_fastgrpo.sh
```

Đổi MODEL_KEY và launcher sang `qwen25_3b` hoặc `qwen3_4b` tương ứng.
Tất cả 14 train wrappers vẫn có cặp OPD/FastGRPO và env overrides.
Defaults giữ: SDPA, target/draft LR 1e-5, batch8, accumulation4, responses8,
draft token budget2048, OPD rank8/Top16, fastLR0.01, update_stream0, profile0,
diagnostics0, proposal/dense implementation auto.

## Validation và benchmark

Raw KV result: `reports/opd_20261007_kv_rtx3090.json` (24 measured cases: 4 lengths
× 3 batches × 2 implementations). BF16, one KV layer, heads4/dim64, median25 warmed
CUDA-event samples; finish event một lần sau warmup. Peak gồm input/scratch và
append length+8 qua geometric-growth boundary, không phải chỉ KV pool bytes.
`previous_fixed_pool` reproduces previous OPD reserve/remap, KHÔNG phải historical
FastGRPO end-to-end baseline. Batch64:

| History | KV trước finish MiB cũ → mới | Peak MiB cũ → mới | Finish ms cũ → mới |
| --- | --- | --- | --- |
| 256 | 138.5 → 16 | 222.5 → 68.75 | 0.114 → 0.075 |
| 512 | 234.5 → 32 | 374.5 → 132.75 | 0.194 → 0.116 |
| 1024 | 426.5 → 64 | 678.5 → 260.75 | 0.354 → 0.195 |
| 2048 | 810.5 → 128 | 1286.5 → 516.75 | 0.670 → 0.351 |

Append median ở case2048 là 0.0574 → 0.0594ms (không claim append speedup).
Finish KV reallocations: 1 → 0 mỗi layer. Live-prefix copy vẫn cần khi row remap.
Các counters bao gồm cả warmup/repeated measurements, không phải một training run.
Script B200 xuất `outputs/benchmarks/kv_<model>_<timestamp>/{target,draft}/`
`report.json` và `kv.csv`. Frozen sweep report thêm cache bytes, host syncs/round,
full reallocations/history copies/bytes; opt-in replay thêm KV compaction time.
Baseline counters không instrument là null, không ghi giả bằng0.

Xem section đầu IMPLEMENTATION_REPORT.md. Máy kiểm tra là RTX3090, không phải
B200. Production model/data/pretrained checkpoint không có tại máy này; chưa có
FastGRPO-vs-OPD AAL/tokens/s hoặc B200 crossover thực. Không tự launch full training.
Tuning component JSON chỉ là synthetic fixture và không được deploy sang B200.
