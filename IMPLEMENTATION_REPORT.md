# Fair persistent training and shared execution (2026-10-07)

This revision supersedes older "historical timing baseline / OPD-only infrastructure"
and persistent SpecForge TTT/KL/LK descriptions below. Scope is the requested shared
training regime and generic execution infrastructure, not new proposal optimizations.

Objective before: SpecForge OnlineEagle3Model TTT/unrolled KL/LK, with simulated
acceptance returned in the second legacy loss slot. Objective now, BOTH methods:
`2*SmoothL1(next_feature[:-1], final_target_H[1:]) + 0.1*soft-label CE`.
EAGLE3 inputs remain original three-layer3H features, but prediction/feature labels
are H, using the saved final target hidden state rather than comparing H against3H.
Prompt/padding excluded; mean hidden-dimension feature error and valid-generated
position normalization per response, then response averaging as upstream local
FastGRPO implementation. Distribution uses detached full teacher softmax, existing
fixed compact-to-target mapping, compact mass renormalization and student FP32
log_softmax. Head work chunked only at supervised positions, no target-transformer
forward. No OPD A/B terms enter persistent loss. SpecForge architecture/weights and
pretraining remain unchanged; online persistent TTT is replaced by teacher forcing
because additional unrolled objectives would not equal the requested upstream loss.
Checkpoint format remains specforge_eagle3_fastgrpo_v1; optional A is absent/off
for FastGRPO and preserved for OPD. Existing pretrained weights remain loadable.

Defaults: paired/generic launchers target/draft **1e-5/1e-5 → 1e-6/1e-4**;
common draft LR **1e-6 → 1e-4**; parser already had1e-6/1e-4 and is verified.
Draft accumulation default1 everywhere; OPD_FAST_LR0.01 unchanged and not confused
with persistent draft LR. All env/CLI overrides retained; no dataset/model/path changes.

Execution: BOTH modes now use the same speculative_generate loop, persistent /
growable target+draft KV, minimal-move remap with canonical sampler ordering,
reused attention/position/tree/path/scheduling buffers, contiguous history and
accepted-suffix in-place compaction. Tree buffer allocation extracted once into
helper/shared_rollout.py and used by both runtimes; contiguous confidence prefix
view preserves native historical confidence-TopK input strides without copies.
FastGRPORuntime proposes raw FP32 softmax + native torch.topk(exact draft_k), and
retains historical native confidence TopK. OPD keeps its previous Top16/correction
and tied-candidate/parent-closure semantics; they are NOT imposed on FastGRPO.
Same budget/adaptive configuration and unchanged sample-once verifier. No extra
target/draft transformer forward. Frozen historical implementation retained only
as external correctness reference, not the default benchmark timing baseline.

FastGRPO constructs no OPD runtime, A/B, teacher Top16/union, correction dispatcher,
KL/update stream. Training constructs its EAGLE adapter with opd_rank=None, so even
unused A parameters/grad buffers are absent. Shared code never calls OPD teacher
capture in baseline mode. OPD B/A objective and update code/stream behavior unchanged.

Added rollout CSV fields for both modes: iter_draft_feature_loss,
iter_draft_distribution_loss, iter_draft_total_loss. These are weighted2.0/0.1
components and their sum, not simulated acceptance. Existing scalar loss transfer
is reused; no logging synchronization. Shared KV/host counters and benchmark labels
now describe fair shared baseline, not old unoptimized historical timing.

Validation: **501 passed, 3 skipped, 3 failed** in the full local suite; targeted
benchmark/launcher tests also pass after summary-counter compatibility adjustment.
Failures are unchanged missing offline-install assets (requirements-bootstrap.txt,
requirements-external.txt, scripts/build_offline_wheelhouse.sh), not hidden/disabled.
Direct formula/compact support/masks/gradients tested against independent reference;
no teacher gradients. Ragged batched vs per-response actual EAGLE training, inference
tensor conversion, all14 paired launcher defaults/overrides, actual adapter no-A /
checkpoint roundtrip pass. Historical/shared FastGRPO tokens, RNG, accepted counters
and original-order feature histories match on CPU/CUDA and native tiny Qwen2 +
real vendored EAGLE3. Physical row-order differences from swap-remove are validated
with per-row tree-mask equality, not mistaken for topology changes. No extra model
forward tests and cold OPD distribution invariants preserved. Compileall, shell
syntax, source check, diff check and isolated pip check pass. Local hardware remains
RTX3090/Torch2.5.1+cu124/Triton3.1/native HF5.12.1 with the previously documented
test-process-only Dynamo alias. No production dependency changes/certification or
full B200 training/benchmark claim.

Changed files: new helper/{shared_rollout,eagle3_online_objective}.py;
grpo_speculative.py; helper/{specualtive_generate,opd_reflex,opd_scheduling,
eagle3_specforge,rollout_metrics}.py; shared B200 env and all14 train wrappers;
scripts/{run_fastgrpo_fair,run_opd_reflex,benchmark_online_draft_training}.sh,
scripts/{benchmark_opd_reflex,check_training_sources}.py, launch/train_model.sh;
fairness/EAGLE/shell/import fixture tests; README, run guide and this report.
No edits to historical_fastgrpo.py, target sampling, verifier, OPD feedback/correction
kernels, reward/GRPO objective, pretraining code/dependencies or sibling repositories.

---

# Last three OPD bottlenecks — minimal moves / persistent KV / compact teacher (2026-10-07)

Current commands/details: [OPD_MEMORY_LAST_THREE.md](OPD_MEMORY_LAST_THREE.md).
Scope restricted to the requested3 memory bottlenecks and production stream default.

Implemented:

- Tail-to-hole disjoint swap-remove plan from the EXISTING host scheduling packet.
  One in-place K/V copy kernel/layer touches moved_rows*live_history, no survivor-wide
  gather. Tail finish = zero KV copy/launch. Both target and EAGLE caches use the
  same plan; owners/responses/lengths/masks/positions/committed features/history
  remap consistently. Generic arbitrary-permutation cache API remains for prefill
  repetition/tests; finished rollout no longer uses it.
- Preserve native target sampling order: target head INPUT hidden states are
  gathered to canonical original-response order only when swap actually changes
  order. No full logits/probs permutation or extra head/transformer forward.
  Small tree/path metadata remap around the UNCHANGED verifier (one trace).
  Canonical feedback queue reads physical draft caches through a row map, retaining
  prior reduction/gradient ordering. This compatibility work is necessary to avoid
  RNG changes from physical batch-row reordering, not a new decoding algorithm.
- `model._opd_target_kv_pool` / `_opd_draft_kv_pool` persist capacities across
  DataLoader iterations. Logical length/B reset at begin/end, stats reset per iter,
  no stale history exposed. Only actual token/batch high-water overflow grows.
  `OPD_KV_MAX_RETAINED_TOKENS=0` retains high water; optional positive cap drops
  oversized pools at rollout boundary. No empty_cache/sync just for logging.
  Persistent pools stay allocated during GRPO training; cap can trade reuse for
  lower retained VRAM if needed. No unlimited growth with fixed bounded workload.
- TargetTop16/mass + p_target_at_DraftTop16 captured while sampler probabilities
  and existing sort are alive. Raw Draft p lookup FUSED into sorted/full-greedy/
  compact-teacher-merge kernels, no standalone lookup launch/second sort/softmax.
  Async feedback receives only small independent preallocated metadata buffers,
  no [N,V] target probabilities or record_stream on them. Union/KL/grad_B/grad_A
  preserve previous normalization/math. Legacy full-prob APIs retained only for
  before/after oracle and parity tests, not production generation.
- Production `OPD_FAST_LR=0.01`, `OPD_UPDATE_STREAM=1` in common env, launcher,
  Python CLI/direct rollout default and all14 paired wrappers. Overrides retained.
  Async snapshot one round stale; NO wait merely for exact dispatch count.
- Per-iteration CSV / frozen benchmark log moved rows, history copy bytes, growth
  reallocations AND initial pool allocations, memory, host packets/round, AAL and
  verification rounds. Rows counted logically (not multiplied by target/draft or
  layers); bytes sum K+V payload across all layers, including necessary prefill
  replication and geometric grow. Tail-finish zero-copy is a per-finish delta.
- Opt-in `--include-previous-opd` runs frozen prior rollout (stable survivor order,
  per-rollout KV and full-prob async feedback) beside current and historical OFF.
  Same policy/draft/A/data/seed/tree/proposal config, separate warmup for each;
  recommendations choose CURRENT configs only. Frozen oracle is imported only by
  tests/explicit benchmark, never by training. Whole-kernel fingerprint changed;
  existing tuner/discovery algorithms untouched, regenerate compatible profiles.

Validation:

- Full suite: **491 passed, 3 skipped, 3 failed**. All new tests pass. The three
  failures are pre-existing references to missing requirements-bootstrap.txt,
  requirements-external.txt and scripts/build_offline_wheelhouse.sh. No tests
  hidden/disabled and no claim that the complete suite is green.
- Tail-only/first-hole/multiple-hole/all-finished CPU/CUDA cache equality and
  copied-byte counts; stable KV pool pointers, grow/reuse/reset/retention cap.
  Compact metadata full/subset mapping, greedy/sample, KL/counters/grad_B/grad_A
  parity against full-prob reference within current tolerances. Weakrefs prove
  full probability tensor dies at sampler return before async feedback.
- Frozen previous vs current complete fixture rollout: tokens, CUDA RNG, history
  tensors, acceptance counters and forward counts match. Native HF Qwen2 + actual
  vendored SpecForge EAGLE3 also passes previous/dynamic/persistent/reuse runs for
  stream0/1. Actual benchmark-driver GPU smoke runs historical/previous/current
  with those tiny real models; only external loading/tokenization/data are test
  fixtures, GPU forwards/sampler/verifier/OPD/measurement/aggregation are REAL.
- Actual RTX3090 tiny fixture per iter: finish rows copied **49 → 5**, history
  copy bytes **64736 → 8672**, pool allocations iter0/1/2 **2/2/2 → 2/0/0**;
  AAL1.212 and one host packet/round unchanged. This is correctness/instrumentation
  evidence, NOT B200 production throughput or production memory evidence.
- Compile/shell syntax/source/CLI checks and local isolated `pip check` pass.
  Tests use RTX3090, Torch2.5.1+cu124/Triton3.1/native HF5.12.1 with the documented
  TEST-PROCESS-ONLY Dynamo alias for vendored SpecForge. No dependency changes.

**Not measured here:** B200 production before/after generation tokens/s, peak
VRAM, AAL/rounds. B200 and server model/data/checkpoints unavailable. User-provided
1837.201 tokens/s and +0.00980 AAL for LR0.01/stream1 are prior sweep evidence only,
used to set requested default, not fabricated/rebranded as our after result.
The documented same-config B200 before/after command exports all requested metrics.

Changed files: `helper/{opd_static_cache,opd_kv_kernels,opd_sampling,opd_reflex,
opd_reflex_kernels,specualtive_generate,rollout_metrics}.py`;
`scripts/{launch/train_model.sh,benchmark_opd_reflex.py}`; shared B200 env,
`grpo_speculative.py` stream default only; all14 paired train wrappers;
tests/fixtures + frozen `tests/oracles/opd_rollout_before_memory.py`;
`OPD_MEMORY_LAST_THREE.md`, run guide/autotuning-doc cross-reference and this report.
Historical FastGRPO, shared target sampler, verifier, proposal autotuner/dispatcher
files, target/draft loss/update schedule, dependencies and sibling repos unchanged.

---

# Generalized OPD autotuning — execution key / six-model dedup (2026-10-07)

Current procedure: [OPD_AUTOTUNING.md](OPD_AUTOTUNING.md), also integrated into
[huongdanchay.md](huongdanchay.md). This section supersedes older model-named
profile paths, fixed trial grids and single-crossover dispatch descriptions.

Implemented:

- `bash scripts/tune_opd_proposals.sh` defaults to all6 requested Qwen models,
  regardless of a previously exported training MODEL_KEY. `OPD_TUNE_MODELS` selects
  any explicit subset/custom model. Read real draft config, checkpoint shape/dtype/
  learned-projector rank and validate fixed d2t/t2d. Torch meta/mmap / safetensors
  headers avoid loading full model weights onto GPU. Missing/corrupt resources
  warn+skip; no valid resources means no measured profile fabricated.
- Deduplicate by GPU name/cc + V/rank/execution dtype/TopK + proposal+merge hash +
  Torch/Triton/CUDA versions. Hidden size and model names are NOT keys. Proposal
  benchmark directly consumes raw logits and projected U, excluding common
  H-dependent feature projection. Each unique config measured once; existing exact
  profiles reused. tqdm shows inspected models/configs/contexts/active-row trials.
- Context grids generated geometrically from actual batch*responses*max_draft_k;
  slots generated from actual V. No absolute model-specific context/slot tables.
  JSON stores measured sparse/fused/GEMM costs plus threshold/dense summaries and
  execution metadata. Dispatcher interpolates measured costs in log(contexts)
  and linear active rows, clamps outside range, memoizes host decisions. Valid
  profiles never consult the uncalibrated global fallback. Exactly one correction
  backend launches. No new device scalar read, kernel or transfer for dispatch.
- Startup exact discovery in launcher before model loading and direct runtime/
  sweep setup. Log mode/profile/GPU/cc/V/rank/dtype/hash; wrong explicit profiles
  reject instead of silent reuse. Missing auto profiles warn and use the existing
  explicitly UNCALIBRATED safe fallback. `OPD_AUTO_TUNE_IF_MISSING=1` opt-in only;
  default0 never surprises a training job with long tuning. Runtime re-discovers
  and rebuilds dtype-sensitive buffers if dtype/vocab/device configuration changes.
- One existing scheduling D2H packet per round retained. Stream0 reads exact
  post-update active_count. **Per user's explicit choice**, stream1 preserves
  overlap and reads stable pre-update snapshot (one round stale); no wait merely
  for dispatch. Kernel still consumes CURRENT B/active set with safe sparse bound:
  dispatch lag can affect performance, not numerical results/update semantics.
- Rollout CSV adds fused/gemm counts and retains sparse/dense/active_rows fields;
  step CSV/JSONL and frozen sweep summaries include new backend counts. Fused/GEMM
  counts use already-known host selection, NO added GPU atomics/logging kernels.
  Legacy dense counter stays unchanged. Resume CSV schema migration preserved.

Validation:

- Full suite: **462 passed, 3 skipped, 3 failed**. All new tuning/discovery/dispatch
  tests pass. Three failures are the same pre-existing absent offline-install
  assets: requirements-bootstrap.txt, requirements-external.txt,
  scripts/build_offline_wheelhouse.sh. Tests not hidden/disabled; no all-pass claim.
- Real CUDA proposal tuner smoke measured two tiny fixture checkpoint/configs of
  different hidden size with one shared execution profile; sparse/fused/GEMM
  probability/ID/normalizer parity bitwise. Fixture profiles remain temporary test
  artifacts, not production measurements. Pure tests cover V41/127/521, ranks4/8/16,
  contexts/active counts, mismatched GPU/cc/V/rank/dtype/hash/compiler/TopK, explicit
  failures vs automatic skip, corrupt checkpoints, exported safetensors and saved
  projector, dedup/reuse, startup resolver and execution-shape re-discovery.
- GPU tests prove only selected correction preparation launches and all three
  counters register when a test profile requests them. One packet per round / no
  production synchronization tests preserved. Actual native HF Qwen2 + vendored
  EAGLE3 complete rollout smoke still passes stream0/1, with unchanged RNG/histories
  and exact expected target/draft forward counts.
- Compileall, all shell syntax, launcher/source checks and diff checks pass.
  `pip check` reports no broken requirements in isolated RTX3090 validation venv
  (Torch2.5.1+cu124/Triton3.1/native HF5.12.1; test-process-only Dynamo compatibility
  alias as documented below). Production dependencies NOT modified/certified.

**Not measured:** production B200 six-model tuning and end-to-end throughput/AAL
before/after. Server checkpoints/data and B200 are unavailable here. Frozen
end-to-end benchmark commands are supplied; no fabricated result, no claim of
guaranteed speedup or zero regression across hardware without those measurements.
Historical FastGRPO generation, target sampler/RNG, verifier, objective/B update
and online draft loss/schedule untouched.

Changed files: new `helper/opd_profiles.py`, `scripts/resolve_opd_profile.py`,
`tests/test_opd_profiles.py`, `OPD_AUTOTUNING.md`; modified
`helper/{opd_reflex,opd_reflex_kernels,opd_scheduling,specualtive_generate,
rollout_metrics,step_metrics}.py`, `scripts/{tune_opd_proposals.py,
tune_opd_proposals.sh,launch/train_model.sh,sweep_opd_reflex.sh,
benchmark_opd_reflex.py,check_training_sources.py}`, shared B200 env, shell/profile
revision tests, run guide/optimization notes and this report. Sibling repos and
unrelated existing local reports left unchanged.

---

# Launcher fix — optional variables under set -u (2026-10-07)

`scripts/launch/train_model.sh` now sets optional defaults after sourcing configs,
before their first use: rollout flush interval1, optional projector LR/profile
empty, proposal/dense implementations auto. Older/custom configs leaving those
variables unset no longer trigger an unbound-variable error. Explicit overrides
are preserved. Invalid/nonpositive flush intervals fail clearly before training.

Validation: launcher/source-import tests **46 passed**, including all14 model/method
launchers with the five options deliberately unset, override preservation and
invalid-value errors. Shell syntax, Qwen3-1.7B dry-run and diff checks pass.
No model/data paths, training algorithms, sampling, dependencies or GPU kernels
changed in this launcher-only fix. No server GPU training was launched.

---

# OPD memory completion — growable KV / mask reuse / compact metadata (2026-10-07)

Current code and commands: [OPD_OPTIMIZATION_20261007.md](OPD_OPTIMIZATION_20261007.md).
This section supersedes fixed-capacity descriptions and earlier validation counts.

Implemented target + EAGLE growable KV: initial256/real prompt rounded to256,
geometric growth only when needed, suffix writes/crop/accepted suffix in-place.
Batch axis reserves actual responses; repeat/select reuse same pool and shared
live-prefix CUDA gather/scatter workspace, not a new full-capacity allocation.
Weak owner references prevent rollout-end GPU storage being held by cyclic GC.
Growable causal/tree/expansion mask and position workspaces replace large per-round
mask allocations. Target sampler consumes its existing sort inside a callback;
selected teacher Top16/mass buffers own storage and full sorted arrays are released
at sampler return. Full probs remain only for exact DraftTop16 probability lookup.
The temporary teacher alias is dropped after feedback enqueue (stream-safe), so
old full probabilities do not overlap the next sampler's arrays. Whole-tree
activations and transient KV views are released as soon as gathering completes.
No second sort, extra transformer forward, target distribution/RNG/verifier change.
Learned A, rollout-local B, selected-only updates, adaptive proposal dispatch and
async update waiting only before the next reader are preserved. Baseline unchanged.

Instrumentation: one scheduling D2H packet/verification round, grow-only pool
reallocation counters, honest full-history/live-prefix copy counts/bytes, cache and
workspace bytes. Per-iteration CSV keeps its previous fields and adds these host
counters. Older CSV headers migrate with a backup on resume. Per-GRPO-step CSV/JSONL
remain unchanged. Profiling stays opt-in; no production cuda.synchronize added.

Validation in isolated local venv (RTX3090, Torch2.5.1+cu124/Triton3.1,
native Transformers5.12.1; production dependencies unchanged):

- Full suite: **411 passed, 3 skipped, 3 failed**. All new KV/mask/sampler/rollout
  tests pass; three pre-existing failures reference absent offline-install assets:
  `requirements-bootstrap.txt`, `requirements-external.txt`, and
  `scripts/build_offline_wheelhouse.sh`.
  These assets are absent in HEAD as well, not caused by this revision. They were
  not fabricated and tests were not disabled to make a green summary.
- Actual tiny HF Qwen2 + vendored SpecForge EAGLE3 complete GPU rollouts compare
  dynamic vs growable cache, stream0/1: identical tokens, accepted counters,
  feature/history tensors and CUDA RNG; exact expected target/draft forward counts.
- CPU/CUDA growth/crop/repeat/select/permutation/duplicate-row tests; no pool
  replacement on finish, no cyclic owner retention; no KV history cat in hot loop;
  exact reused masks; sampler FP32/BF16/FP16 tokens/probs/RNG and sorted-array
  lifetime; positive teacher ties/nonzero support; runtime extent kernel contracts.
- `python -m compileall -q .`, all shell syntax checks, paired launcher dry-run,
  benchmark CLI validation, source integrity and `git diff --check` pass.
  `python -m pip check`: no broken requirements in the isolated local venv;
  this is NOT validation of the unavailable B200 production stack. SpecForge
  imports use a test-process-only alias of Torch2.5 `cache_size_limit` to its newer
  `recompile_limit` name. No production import/compile fallback was added.

Measured raw component data: `reports/opd_20261007_kv_rtx3090.json`, all24 cases
successful. Batch64/history2048/one BF16 layer heads4/dim64: old fixed pool vs new
growable KV before finish **810.5 → 128 MiB**, peak **1286.5 → 516.75 MiB**, finish
compaction **0.6697 → 0.3509 ms**, finish reallocation **1 → 0**. Geometric growth
and exceptional row remap still copy live prefixes; finish latency remains linear
in surviving history. No claim of O(new suffix) for exceptional row remap.
Append **0.0574 → 0.0594ms** in that case, so no append speedup claim.

**Not measured:** B200 full-model peak VRAM, AAL, generation tokens/s/utilization
before/after and long production-rollout stability. Server assets are unavailable.
Run `scripts/benchmark_opd_kv.sh` and the frozen sweep on B200; component numbers
are not end-to-end evidence. Existing proposal profiles must be regenerated after
kernel fingerprint change; do not deploy RTX3090 fixture profiles to B200.

Files changed/added this revision:
`helper/{opd_static_cache,opd_kv_kernels,opd_attention,opd_attention_kernels,
opd_sampling,opd_scheduling,opd_reflex,opd_reflex_kernels,specualtive_generate,
rollout_metrics}.py`; `scripts/{benchmark_opd_kv.py,benchmark_opd_kv.sh,
benchmark_opd_reflex.py,check_training_sources.py}`; OPD/EAGLE tests/fixtures;
this report, optimization notes, run guide and measured KV JSON.
`.gitignore` whitelists only that measured JSON; other local reports stay ignored.

---

# Prior OPD follow-up — fixed KV / sampler reuse / telemetry (historical notes)

Current behavior and commands: [OPD_OPTIMIZATION_20261007.md](OPD_OPTIMIZATION_20261007.md).
This section supersedes the older implementation/validation description below.

Implemented fixed-capacity target + EAGLE draft KV append/crop, reused sorted target
sampler intermediates for full-vocab/permutation teacher Top16, direct greedy teacher,
compact sparse correction scratch, selected-only head/union/B/A feedback work,
tree candidate/branch/mask/packed metadata pools, production-config tuner/autoload,
and optional separate projector LR with optimizer-state migration. No historical
FastGRPO generation/sampler or dependency versions changed. Profiling stays off.

Added buffered `rollout_timing.csv`: one row per executed DataLoader iteration,
including reward-filtered/no-training/invalid batches, weighted cumulative AAL,
monotonic iterator state restored from checkpoint, per-rank files without logging
all-reduce, host-only telemetry snapshot (no retained GPU histories). End-iteration
wall clock includes setup and checkpoint work. Flush on checkpoint/end/interrupt.
Existing GRPO-step CSV/JSONL remain; all 14 paired launchers expose projector LR
and rollout flush interval.

Validation:

- Full suite: **387 passed, 3 skipped, 3 failed**. The three failures are existing
  `tests/test_requirements.py` references to absent `requirements-bootstrap.txt`,
  `requirements-external.txt` and `scripts/build_offline_wheelhouse.sh` (also absent
  in HEAD before these edits). Those obsolete offline-install assets were not
  fabricated/reintroduced or their tests disabled. No claim of full-suite success.
- CPU/CUDA static-vs-dynamic rollout equality: generated tokens, histories, RNG,
  target/draft forward counts, stream0/1. Native Transformers **5.12.1** tiny
  Qwen2/Qwen3/Llama logits match bitwise on CPU/CUDA, eager and SDPA, including crop.
  Actual vendored SpecForge EAGLE layer/adapter static vs list cache matches in
  FP32/BF16 on CPU/CUDA. Teacher permutation/positive ties/zero probability and
  greedy match reference; no fallback vocabulary extraction invoked in those tests.
- Sparse/fused/GEMM probability/ID parity, compact scratch reuse, historical golden,
  selected projector gradient, LR/checkpoint moments, CSV skip/resume/weighted ratios,
  one scheduling packet/round, compileall, shell syntax, paired CLI dry-run, training
  source integrity and `git diff --check` checked. `pip check`: no broken requirements
  in the isolated local validation venv, NOT a production B200 stack certification.
- Hardware: RTX3090; Python3.10, Torch2.5.1+cu124, Triton3.1. Transformers5.12.1
  installed only in a temporary system-site-packages venv, not the user's base env.
  SpecForge tests used a **test-process-only** `torch._dynamo.config.recompile_limit`
  compatibility alias because Torch2.5 calls it `cache_size_limit`. No production
  source workaround or dependency downgrade was added. Production pins untouched.

Component measurements are stored in `reports/opd_20261007_rtx3090_proposal.json`:
synthetic BF16 V32768/H128/rank8, context workloads 1/8/64, active slots
0/16/256/4096/V, median25 warmed CUDA-event samples per mode. All 15 cases passed
bitwise proposal parity. This is **not production shape, B200 crossover, AAL, or
end-to-end speedup evidence**; production tuner requires the actual draft config.
Only the chosen backend launches; profiler/timer events exist in benchmark code,
not production proposal dispatch.

Remaining boundaries, explicitly:

- Host sync/verification round: **1 before → 1 after**, plus setup/end-rollout
  transfers; cannot claim a fully GPU-resident decoder.
- Teacher full-vocab rescans removed for ordinary full-map sorted-sampler inputs.
  Positive boundary ties require reading the tied interval (uniform worst case:
  whole support); subset mappings and unfiltered sampling still use a selected-state
  compact scan. No extra target softmax/sort/model forward.
- No full-prefix KV copies on append/accept/crop rounds. Prefill repetition and
  finished-batch compaction still copy surviving prefixes. Static conservative
  capacity increases reserved VRAM; production peak memory is not measured here.
- Small tree native TopK/gather still allocate internal scratch; not zero-allocation.
  Gradient-A selected reduction changes FP32 association; mathematical semantics
  preserved/tested with tolerances, universal bitwise trained-trajectory replay
  across hardware/revisions is not promised.
- **B200 FastGRPO vs OPD AAL/tokens/s, GPU utilization, long-response memory and
  crossover remain unmeasured**: B200 and server model/data/checkpoints absent.
  Run the documented tuner and frozen-rollout sweep before selecting LR/stream or
  claiming speedup. No full training was launched.

Changed files: `grpo_speculative.py`; `helper/{eagle3_specforge,opd_reflex,
opd_reflex_kernels,specualtive_generate,tree_verification,opd_scheduling}.py`;
new `helper/{opd_static_cache,opd_sampling,opd_optimizer,rollout_metrics}.py`;
`scripts/{tune_opd_proposals.py,tune_opd_proposals.sh,sweep_opd_reflex.sh,
launch/train_model.sh}`; shared env and 14 `train_*.sh`; OPD/EAGLE fixture/tests;
run guide, this report, new detailed optimization document and benchmark JSON.

---

# Previous OPD performance revision — 2026-10-07 (historical notes)

This section supersedes the older sparse-only defaults / dual-launch adaptive
description and historical validation numbers below. Historical FastGRPO's
generation, sampling, history, and model algorithms were not edited.

Implemented:

- Runtime extents/strides in tree/OPD Triton kernels use `do_not_specialize` rather
  than constexpr. Tree mask columns use fixed 256-wide tiles. A GPU test changes
  past=3/17/1025/2051 and live batch/rows without increasing compiled variants.
  Power-of-two block buckets and fixed model vocabulary/rank still specialize.
- Default `OPD_PROPOSAL_MODE=auto`. Exactly one correction backend launches.
  Interpolated log(contexts) crossover handles unseen shapes; nearest measured
  dense-implementation bucket chooses fused vs tiled GEMM. No profile means an
  explicitly uncalibrated V/8 threshold and fused dense. Profiles reject a different
  GPU/compiler/kernel. No B200 threshold is invented or shipped as measured.
- Fused dense correction + Top16 + normalization avoids writing/reading dense
  corrected logits. Alternative GEMM uses preallocated workspace. One Top16
  serves both OPD and tree TopK. Ordered rank adds and explicit pairwise FP32
  normalization prevent layout-dependent sparse/fused-dense rounding differences.
  This normalization has the same mathematical semantics but may round differently
  from the previous generic reduction; old checkpoint trajectory bitwise replay
  across this code revision is not promised.
- Active bitmap/IDs are append-only during rollout. `_round_end` is O(1), with
  GPU sum/max counters; it no longer loads/scans B or compacts zero rows.
- Teacher visited/frontier IDs compact on GPU before a persistent selected-row
  vocabulary scan. Unselected rows do not read target probabilities or run Top16
  reduction. Existing post-sampling target probabilities are reused; no extra
  target sort/softmax or target/draft transformer forward.
- OPD history has contiguous original-response pools, chunk index_copy with GPU
  owner IDs, finish-length metadata only, and one compact final gather per field.
  No per-response clone at finish or per-round full-history concatenation. Rare
  capacity doubling handles verification padding; no old pool survives that grow.
- Side-stream waits remain immediately before the next proposal. Dispatch's GPU
  active-count snapshot piggybacks on the existing scheduling packet and can lag
  one feedback update. Actual correction always uses CURRENT GPU state. This is
  only a performance choice, not stale correction or feedback gating.
- Small union/dedup/p/q/tail/KL/gradient terms remain fused on GPU. Learned A
  manual gradients and the existing optimizer/DDP boundaries are unchanged.
- Added step/cumulative active_rows_mean/max and sparse/dense_rounds aliases to
  both CSV/JSONL. Step maximum is an interval maximum, NOT a difference of running
  maxima. Repeated optimizer updates with the same step label merge their maxima.

Host synchronization per OPD verification round: **1 before → 1 after**, the
existing consolidated HF scheduling packet. No extra transfer for backend dispatch,
teacher selection, or metrics. Profile-OFF introduces no cuda.synchronize. Prefill
setup/end-of-rollout returned lists and packed metrics still have boundary transfers;
the entire decoder is not claimed zero-sync. Optional profiling is separate.

Validation on RTX3090, Python3.10 / Torch2.5.1+cu124 / Triton3.1:
260 related tests passed, 3 CPU-only stream parameterizations skipped; one existing
torch.load FutureWarning. Covers historical golden outputs, no extra forwards,
all teacher support cases, projector/checkpoint, both update streams, history vs
concat, exact per-step telemetry, one packet/round, one chosen correction backend,
compiled variant reuse. compileall, shell syntax, CLI dry-runs, source integrity,
git diff --check and pip check pass in the existing local test environment.
Production B200 pinned environment was not installed here.

Isolated proposal benchmark (BF16, V32768, H2048, rank8, median of 10 CUDA-event
samples): 7 context workloads × 8 active counts × sparse/fused-dense/GEMM/auto.
All Top16 IDs/probabilities/normalizers passed BITWISE cross-backend parity.
Raw evidence: `validation/opd_performance_20261007/rtx3090_proposal.json`.

| Contexts | Sparse ms, S=V | Fused dense ms, S=V | Auto ms, S=V | Measured threshold |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2464 | 0.2147 | 0.2167 | 0 |
| 8 | 0.2659 | 0.2161 | 0.2190 | 0 |
| 32 | 0.4136 | 0.2375 | 0.2401 | 0 |
| 64 | 0.4752 | 0.3178 | 0.3204 | 0 |
| 128 | 0.6523 | 0.4940 | 0.4965 | 4096 |
| 256 | 1.0105 | 0.8539 | 0.8632 | 8192 |
| 512 | 1.7881 | 1.6128 | 1.6134 | 8192 |

Threshold 0 means dense is preferable at all measured nonzero counts in that workload; cold
still skips correction. Auto timings use the uncalibrated V/8 policy, not an
artificially perfect switch. At 512 contexts fused dense was 1.5862 ms at S4096
and 1.6128 ms at S32768; sparse was 1.5857→1.7881 ms. These are component results
on RTX3090, not before/after end-to-end speedup or B200 crossover measurements.

B200 end-to-end FastGRPO vs OPD AAL/tokens/s: **not run**. Actual Qwen model,
pretrained EAGLE checkpoint, and configured dataset are absent on this machine.
No synthetic AAL replaces this experiment. `scripts/sweep_opd_reflex.sh` runs both
methods with the same frozen resources/seeds and exports AAL, round counts,
generation wall/tokens/s, VRAM, active rows, backend counts, optional nvidia-smi
utilization, and separate optional profiling replay. Instructions in huongdanchay.md.
Positive delta AAL is a measurement goal, not an implementation guarantee.

Remaining costs: one scheduling packet; target top-p sampling; one compact scan
per selected teacher state; selected lm_head row reconstruction; learned-A GEMM;
HF DynamicCache's native append and finished-batch KV index_select. Accepted KV
suffix compaction was already in-place and remains so. No unvalidated static-HF
cache replacement was introduced. Contiguous history reserves capacity up front
and therefore may use more memory early in rollout than the former small per-row
buffers; final packing temporarily needs both pool and used output storage.

Changed files: helper/{opd_reflex,opd_reflex_kernels,tree_kernels,opd_scheduling,
specualtive_generate,step_metrics}.py; new helper/opd_history.py;
grpo_speculative.py; shared config, scripts/launch/train_model.sh and 14 paired model launchers;
scripts/{tune_opd_proposals.py,tune_opd_proposals.sh,benchmark_opd_reflex.py,
gpu_utilization.py}; related tests/fixtures and docs. Requirements/model/data paths
and historical_fastgrpo.py/sampling.py/rollout_history.py are unchanged.

---

# Previous revision history (superseded where noted above)

# Learned OPD / historical baseline optimization — 2026-10-07

## User-requested SDPA target default

Previous target default was eager; EAGLE-3 draft runtime and online trainer were
already SDPA. Changed shared config, all14 paired model launchers, direct GRPO
CLI and frozen benchmark CLI to target sdpa, with ATTENTION_IMPLEMENTATION=eager
override preserved. Both methods use the same selected backend. No sampler,
tree, reward, optimizer or draft-training algorithm change in this update.
SDPA is not guaranteed to choose a flash kernel with arbitrary tree masks, or
to be bitwise identical/faster than eager; no B200 speedup claimed.
Native PyTorch FP32/BF16 tests cover cached-prefix/left-padding/branch masks,
forward/backward and blocked-branch isolation on CPU and RTX3090. Native Qwen
Transformers5.12.1/B200 end-to-end smoke still requires server assets.
Modified configs/_shared/b200_common.env,14 train wrappers, grpo_speculative.py,
scripts/benchmark_opd_reflex.py, tests/test_shell_scripts.py, huongdanchay.md;
added tests/test_sdpa_tree_attention.py. Dependencies unchanged.

## Follow-up: confidence tree parent-closure crash

### Second traceback: reusable proposal scratch alias (supersedes tie-only diagnosis)

Server tree_kernels.py line168 confirms the tie fix WAS deployed; the remaining
failure was not an old checkout. Reproduced before this fix with different
root/child/grandchild distributions, even WITH deterministic confidence ties:
root confidence was a view of proposal_q, overwritten by the next expansion.
Earlier same-distribution fixtures masked this lifetime error.

Root beam/confidence history now copies only B*K FP32 values into separately
preallocated storage. Expansion products/gathers already own their storage;
mapped target IDs are materialized and context caches copied before scratch
reuse. No per-vocab cloning, additional transformer forward or CUDA sync.
Added independent snapshot oracle for actual nested draft_generate, batch64,
depth3/5, K3/7/8, changing batch/context layouts, repeated buffer reuse and CUDA.

Tests also exposed two masked-logit edge cases: all--inf scan tiles produced
NaN mass; exhausted Top16 candidates could repeat IDs at q0. Corrected zero
tile mass and explicit merge eligibility, with support1/3/8 across vocab257/521.
Finite-logit normalization and historical baseline code unchanged. Kernel hash
changed: retune any existing adaptive profile, do NOT bypass its fingerprint.
One test assertion compared uninitialized cache capacity by floating equality
(NaN != NaN); it now checks bitwise equality over the SAME entire capacity.

Modified helper/{opd_reflex,opd_reflex_kernels,specualtive_generate,tree_kernels}.py,
tests/test_opd_reflex.py; added tests/test_opd_proposal_lifetime.py; updated docs.
Final focused rerun: **237 passed, 3 skipped, 1 pre-existing torch.load warning**
on CPU and real RTX3090 CUDA (Torch2.5.1+cu124/Triton3.1). Includes original
historical output tests, stream0/1 rollouts, learned-A gradient/checkpoint tests,
independent changing-distribution tree oracle and masked-logit tests. compileall,
shell/source/diff checks and local pip check pass. This is NOT a native Qwen/B200
end-to-end training pass; the server's actual two-step smoke is still required.
Native B200 Qwen training remains untested locally because weights/data absent;
see two-step server smoke instructions. Dependency pins not changed.

Server traceback starts at the retained parent-closure device assertion, not
Triton compilation or a library version mismatch. Reproduced BEFORE the fix on
a saturated native-sampler/OPD fixture (draft p=1, all other probabilities0):
`RuntimeError: confidence-selected draft tree is not parent-closed`.
FP32 products can tie ancestors at p=1 or underflow0; arbitrary native
confidence Top-K can select a child while excluding its ancestor.

Added OPD-only exact lexicographic tree-confidence selection: original FP32
IEEE score bits as the primary key, ascending full node index as the secondary
key. One fused CUDA encoder into reused small int64 tree workspace, then native
integer Top-K; no epsilon, full-vocabulary sort, probability modification,
extra target forward or host sync. Full selection avoids the key pass entirely.
Keep parent-closure assertion. Historical FastGRPO code is unchanged.

Tests include unit probabilities, all ties, signed zero, adjacent FP32 ULPs,
underflow, multiple rows/noncontiguous score views, parent closure,
attention/verifier parity and saturated rollout with responses8 and stream0/1.
Final focused regression suite: 215 passed, 3 skipped (CPU-only stream cases),
1 pre-existing torch.load warning; real CUDA tests ran on RTX3090,
Torch2.5.1+cu124/Triton3.1. Unique-confidence selection equals original native
Top-K, and RNG states remain unchanged. compileall, source-integrity, shell
syntax, diff checks and local pip check pass.
Production Qwen weights/data and B200 are still absent locally; native B200
training is NOT claimed. See huongdanchay.md for server tests/two-step smoke.

Modified: helper/{tree_verification,tree_kernels,opd_reflex,specualtive_generate}.py,
tests/opd_fixtures.py; added tests/test_tree_confidence_selection.py; this report,
METHOD_OPD_REFLEX.md and huongdanchay.md. Dependencies/policy/reward/sampling
configuration not changed by this fix.

This supersedes the fixed-A/shared-baseline implementation at d9766ad.
FastGRPO now dispatches to frozen historical c3f05ad OFF: exact native softmax,
torch.topk(draft_k), original tree/Python verifier/RNG/stacked KV. No OPD engine
or scheduler/cache optimization reaches this baseline.

User explicitly accepted OPD always fused Top16, including cold; Top16 prefix
ties need NOT match historical K. Cold invariant is raw corrected logits = raw,
same full compact softmax distribution, no slow fallback. Mathematical
probabilities validated with FP32 tolerance, not false old-Torch bitwise claims.

## Implemented

- A is an nn.Parameter ON draft_model, head-aligned deterministic basis only
  at initialization; post-norm/head-input representation. Existing A checkpoint
  loaded unchanged. GPU gradient sum/weight accumulated before B_t changes.
  Cold B=0 has zero A gradient but its valid state weights still count.
  Apply at existing draft optimizer boundary before existing DDP sync.
  Pending gradient buffers checkpointed separately PER RANK for training resume.
  Older fixed-A optimizer checkpoints fail with an explicit initialization-only
  migration message rather than silently dropping/misassigning pending feedback.
  Auxiliary evaluation/analysis and frozen benchmark never accumulate A grads.
- Exact adaptive sparse/dense: ordered FP32 rank GEMM with tile B reuse,
  no FMA/TF32 association change. Sparse writes only S scalars; dense writes
  reused current-proposal workspace. SAME scan positions/normalization gives
  bitwise switch outputs. Current proposal scratch O(B*C*V), allocated once,
  not a probability cache of all rollout states. No guessed B200 crossover.
- TargetTop16 positive-only, sentinel -1/0 for absent entries; no arbitrary
  zero-probability target additions/updates. DraftTop16 preserved and tail intact.
- OPD metadata: THREE round transfers reduced to ONE fixed packet; GPU pad
  masks and position cumsums, no per-round Python padding-set walks. Necessary
  HF crop/active sampler batch host boundary remains (not claimed zero-sync).
- OPD KV suffix gathered into reused small per-head scratch, copied in-place
  and cropped. Accepted history prefix never copied after verification; no
  stacked all-layer/full-history concat. Native HF next update creates normal
  contiguous KV before attention. Finished batch copy still remains.
- Native target/draft transformer work unchanged per verification/expansion;
  no extra forwards/backward/full-vocab KL backward. Shared B local/reset,
  no per-response state or round all-reduce. Two reusable dependency events,
  late wait, no new profiling/default synchronization.
- Existing per-step CSV/JSONL definitions preserved; added cumulative/delta
  sparse/dense root-round usage and existing active-row mean.
- All 14 paired .sh export train-projector/mode/profile knobs. Benchmark is
  HISTORICAL FastGRPO vs optimized OPD, no shared Top16 baseline.
  Crossover tuner fingerprints GPU/Torch/Triton/CUDA/kernel and exact geometry.
  Untuned geometries stay sparse, incompatible profiles fail clearly.

## Evidence actually measured (NOT B200/full-model)

RTX3090 / Torch2.5.1+cu124 / Triton3.1 / Python3.10 existing test environment.
Final focused suite:107 passed,2 skipped (CPU stream cases),1 torch.load warning.
Revised 26-test suite
covers actual historical dispatch unchanged/no OPD constructor, positive-only
teacher1/3/8 support, sparse/dense/edge ties/full S oracle and bitwise switching,
GPU adaptive counters, learned A gradient/optimizer boundary/checkpoint,
one-packet scheduling and in-place KV suffix parity, per-rank pending-gradient
checkpoint restoration. Earlier broader focused run:127 passed,2 skipped.

Final full pytest attempt:189 passed,2 skipped,1 torch.load warning,
12 failures/37 setup errors: missing transformers for actual SpecForge native
training/position tests and three pre-existing missing offline packaging files.
These tests retained; whole-suite PASS or pinned-stack native run NOT claimed.
compileall/shell syntax/source integrity/CLI dry-run/diff checks/pip check pass.

Controlled component benchmark at V32768/H2048/r8, BF16,10 samples per median:
Final isolated run: B8,C8,S32768 sparse0.5033ms vs dense0.3363ms;
S8192 sparse0.3649 vs dense0.3363. B8,C1,S32768 sparse0.2642 vs dense0.2423.
All measured sparse/dense values/IDs
bitwise identical. Component speed is NOT generation speed or acceptance gain;
profiles from this GPU are not B200 profiles. Raw JSON/logs in
validation/opd_20261007, authoritative final component profile:
`crossover_isolated_rtx3090.json`. Do not use `crossover_final_rtx3090.json`
(concurrent tests) as a performance profile. The older archived profile also
has a pre-final kernel fingerprint; it is historical evidence, not runtime config.

Configured production target3B config / pretrained / simplelr data absent here.
Actual asset check fails clearly. No B200 model benchmark, full policy/draft
training, fake data/model or fabricated AAL/tokens/s result generated.

## Remaining bottlenecks / limits

One host scheduling boundary required by native dynamic batch/HF crop and exact
sampling; cannot defer it without unverified fixed-batch/RNG changes. Batch KV
compaction and HF next-cache append still copy history when necessary. Full
compact normalization/teacher scan, selected head rows, A gradient GEMM and
side-stream contention remain measurable costs. S may approach V, now exact
measured dense crossover available. Canonical dense GEMM prioritizes exactness;
cuBLAS/TensorCore speed claims require a separate numerical proof.

A initialized from a learned head is NOT itself trained until optimizer steps
occur. Use an OPD-trained checkpoint to measure learned A. Frozen sweep does not
silently train A. Default sparse/stream0/LR0.01 is untuned conservative; no claim
OPD always beats historical FastGRPO. Preserve/report negative AAL or cost>benefit.

## Commands / files

All commands and paths in huongdanchay.md; detailed algorithm METHOD_OPD_REFLEX.md.
Run scripts/tune_opd_proposals.sh on ACTUAL B200, export its real profile,
then sweep scripts/sweep_opd_reflex.sh and validate held-out end-to-end results.

Created/modified:
- `METHOD_OPD_REFLEX.md`
- `README.md`
- `configs/_shared/b200_common.env`
- `grpo_speculative.py`
- `helper/eagle3_specforge.py`
- `helper/opd_reflex.py`
- `helper/opd_reflex_kernels.py`
- `helper/specualtive_generate.py`
- `helper/tree_kernels.py`
- `huongdanchay.md`
- `scripts/benchmark_opd_reflex.py`
- `scripts/check_training_sources.py`
- `scripts/launch/train_model.sh`
- `tests/opd_fixtures.py`
- `tests/test_opd_contracts.py`
- `tests/test_opd_reflex.py`
- `train_llama31_8b.sh`
- `train_llama31_8b_fastgrpo.sh`
- `train_qwen25_14b.sh`
- `train_qwen25_14b_fastgrpo.sh`
- `train_qwen25_1p5b.sh`
- `train_qwen25_1p5b_fastgrpo.sh`
- `train_qwen25_3b.sh`
- `train_qwen25_3b_fastgrpo.sh`
- `train_qwen25_7b.sh`
- `train_qwen25_7b_fastgrpo.sh`
- `train_qwen3_1p7b.sh`
- `train_qwen3_1p7b_fastgrpo.sh`
- `train_qwen3_4b.sh`
- `train_qwen3_4b_fastgrpo.sh`
- `helper/historical_fastgrpo.py`
- `helper/opd_scheduling.py`
- `scripts/tune_opd_proposals.py`
- `scripts/tune_opd_proposals.sh`
- `tests/test_opd_revision.py`
- `IMPLEMENTATION_REPORT.md`, validation/opd_20261007 evidence.
No model/data/checkpoints/venv/sibling code deleted or altered.
