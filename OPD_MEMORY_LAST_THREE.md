# OPD: minimal-move KV, persistent pools, compact teacher feedback

Chỉ sửa3 memory bottlenecks; không sửa autotuner/dispatcher, historical FastGRPO,
OPD objective, target sampler/RNG, verifier hoặc online EAGLE training schedule.
Default mới: `OPD_FAST_LR=0.01`, `OPD_UPDATE_STREAM=1`. Đây là cấu hình bạn báo đã
nhanh nhất trong B200/Qwen2.5-3B sweep (1837.201 tokens/s, delta AAL +0.00980);
không phải kết quả tôi tự benchmark và không phải bảo đảm cho mọi model.

## Implementation / correctness

- Finish plan lấy từ scheduling packet host đã có. Tail live rows (source>=new B)
  lấp holes (destination<new B), disjoint nên một Triton copy kernel/layer, không
  survivor-wide gather scratch. Tail-only finish chỉ shrink B, zero copy/launch.
  Target và draft dùng cùng move plan. Owners, original response IDs, lengths,
  padding, positions, committed features/tokens và history remap cùng nhau.
- Native multinomial vẫn phải nhận canonical original-response order để giữ RNG.
  Sau swap, reorder input hidden states của target head (H-sized, không [N,V]),
  chạy đúng một head như trước. Small tree/path metadata cũng remap; verifier
  không đổi và chỉ trace một lần. Feedback xử lý canonical selected queue, dùng
  row indirection đọc physical draft caches; giữ reduction/feedback ordering.
- `model._opd_target_kv_pool` / `_opd_draft_kv_pool` persist allocated capacity;
  reset length/B/state trước và sau rollout. Stale storage không logically visible.
  Grow chỉ khi tokens hoặc batch thực sự vượt high-water mark. Stats reset mỗi
  rollout; warm reuse kiểm tra cả growth reallocations và initial pool allocations.
  `OPD_KV_MAX_RETAINED_TOKENS=0` default giữ high-water allocation. Set4096 hoặc
  một budget phù hợp để drop oversized pools cuối rollout. Drop có thể khiến
  iteration sau allocate lại: đây là tradeoff memory vs reuse, không free giữa round.
  Pools còn giữ VRAM trong target/draft training; giảm retention cap nếu training OOM.
- Sampler reuse chính probabilities/sort đã có. Capture TargetTop16 IDs/probs,
  compact_mass và raw p_at_DraftTop16 vào independent preallocated small buffers.
  Draft coordinate lookup **fused vào teacher extraction kernel**, không thêm
  standalone lookup launch, second softmax/sort hoặc model forward. Metadata
  capacity theo verification-node bound, GPU-selected entries hợp lệ; không D2H
  để resize theo selected count. Có tối đa32 probability coordinates/state
  (Target16+Draft16), cộng IDs/mass. Không giữ [N,V] hoặc sorted arrays trong async
  feedback / record_stream. Union/KL/grad_B/grad_A giữ math và normalization trước.
- Async vẫn dùng active-count snapshot một round cũ, không wait chỉ để dispatch.
  One scheduling host packet/round như trước; wait B chỉ trước next proposal đọc B.

Counters trong `logs/rollout_timing.csv`:
`iter_kv_rows_moved`, `iter_kv_history_copy_bytes`, `iter_kv_full_reallocations`,
`iter_kv_pool_allocations`, `iter_host_syncs_per_round`, AAL/verification rounds.
Rows moved là logical rows copied khi finish (không nhân đôi target+draft/layers).
Bytes là tổng K+V payload qua mọi target/draft layers, gồm prefill-repeat/grow/
row compaction; không gọi padding/accepted-suffix tiny gather là full-history copy.
Zero-copy tail finish là delta riêng của finish, không có nghĩa cả rollout không
cần prefill repeat copies. Reallocation counts chỉ grow; pool_allocations còn
đếm initial allocation. Timing/metrics definitions và per-step logs giữ nguyên.

## Chạy trên B200

Đồng bộ toàn bộ files sửa, gồm tests/oracles nếu dùng before/after benchmark.
Không cài lại dependencies hoặc pretrain lại.

```bash
cd /workspace/storage-shared/nlp/minhpn19/SpecNaacl
source .venv/bin/activate
export PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES=0
export OPD_FAST_LR=0.01
export OPD_UPDATE_STREAM=1
export OPD_PROFILE=0
export OPD_DIAGNOSTICS=0
export OPD_KV_MAX_RETAINED_TOKENS=0

# Proposal algorithm không đổi, nhưng conservative whole-kernel fingerprint đổi.
# Regenerate compatible profiles, không dùng explicit old-hash profile.
unset OPD_PROPOSAL_PROFILE OPD_TUNE_OUTPUT
bash scripts/tune_opd_proposals.sh

# Frozen same-weight/data/seed comparison: historical, previous OPD rollout,
# optimized OPD. Warm toàn measured seed/prompt schedule để đo steady-state reuse.
MODEL_KEY=qwen25_3b OPD_FAST_LRS=0.01 OPD_STREAMS=1 \
BENCH_SEEDS=42,43 BENCH_ITERATIONS=3 BENCH_WARMUP=2 \
BENCH_MAX_LENGTH=2048 BENCH_MAX_PROMPT_LENGTH=256 \
bash scripts/sweep_opd_reflex.sh --include-previous-opd --gpu-utilization

# Two real GRPO steps trước khi chạy dài.
MAX_TRAIN_SAMPLES=128 bash train_qwen25_3b.sh --max_grpo_steps 2
# Full run chỉ sau smoke:
bash train_qwen25_3b.sh
```

Output: `outputs/benchmarks/opd_qwen25_3b_<timestamp>/{report.json,summary.csv,responses.jsonl}`.
Methods: fastgrpo / opd_reflex_previous / opd_reflex. Summary gồm AAL, verification
rounds, generation wall/tokens/s, peak allocated/reserved, host syncs/round,
KV reallocations/pool allocations/history-copy bytes/rows moved per iteration,
delta AAL và throughput ratio vs previous OPD. Prior rollout snapshot chỉ dùng
opt-in benchmark/tests, không được import trong production training. Cả hai OPD
branches dùng current identical shared proposal/feedback math; snapshot giữ
previous cache lifecycle, full-prob async feedback và stable survivor compaction.
Không lấy fastest previous implementation làm recommendation cho current.

Training output: `outputs/train/qwen25_3b/<run>/logs/`. Model khác dùng paired
train wrappers hiện có; stream0 vẫn override bằng `OPD_UPDATE_STREAM=0`.

## Validation / results

Local RTX3090: full before/after fixture and native tiny Qwen2 + real SpecForge
EAGLE3 rollouts pass stream0/1: tokens, RNG, histories, accepted counters and
transformer-forward counts unchanged. Compact teacher full/subset mapping and
greedy/sampling match full-prob KL/grad_B/grad_A reference in existing tolerances.
No extra host sync; tail-only finish zero-copy; persistent reset/reuse/cap tests.

Actual tiny CUDA fixture (NOT B200, NOT production model/data, no throughput claim):

| Per iteration | Previous | Optimized |
| --- | --- | --- |
| Finish rows copied | 49 | 5 |
| KV history copy bytes | 64,736 | 8,672 |
| Initial pool allocations, iter0 / iter1 / iter2 | 2 / 2 / 2 | 2 / 0 / 0 |
| Growth reallocations | 0 | 0 |
| Scheduling host packets/round | 1 | 1 |
| AAL | 1.212 | 1.212 |

Longer KV unit fixture (601 tokens) separately verifies high-water capacity
reuse with zero reallocations and no stale values in subsequent shorter rollouts.
**B200 production peak VRAM / before-after tokens/s / AAL not measured here:**
server model/data/B200 absent. The user-reported1837.201 is prior evidence only;
run the command above for after-revision measurements. No fabricated speedup.
