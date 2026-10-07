# Learned OPD Reflex / historical baseline — current revision

FastGRPO is historical c3f05ad OFF, native softmax/topk(draft_k), original
tree packing/Python verifier/target sampler/RNG/stacked KV. No OPD optimization
or additional Top16 pass is used in this baseline.

OPD: head_input = h if norm_output else draft.norm(h), u=head_input@A.
A:[H,8] is a persistent nn.Parameter ON THE DRAFT MODEL. Old checkpoints without A
initialize once from a deterministic orthonormal basis of output-head rows,
NOT a random per-rollout/fixed projector. Existing A is loaded, never reset.
Gradient terms v_s=sum_U w_s(q-p)B_t[j] and head_input.T@v accumulate GPU-only,
BEFORE B changes. Apply weighted mean at existing draft optimizer boundary
before existing DDP sync; no round backward/optimizer/all-reduce. Pending
accumulators saved for training resume. Evaluation/frozen sweep does not accumulate
A gradients. A must actually receive optimizer steps to call it trained;
head-basis initialization alone is NOT evidence of learned improvement.

B:[compactV,r] is ONE shared rollout-local FP32 adapter. Visited + expanded,
final-packed one-hop rejected sibling states, no recursive rejected descendants.
Teacher uses existing post-temperature/top-p/top-k probabilities conditioned on
compact mass. Zero/nonfinite compact mass skipped with counter. Target candidate
entry is VALID only if p>0; unused Top16 slots use sentinel-1/prob0 and NEVER add
arbitrary q-0 rows. DraftTop16 remains included even when its teacher p is0.
Union + one tail KL, no union renormalization; q-p exact union-coordinate
gradient with support fixed for the round. No dense tail backward.

OPD always fused Top16→prefix draft_k, including cold. Cold corrected logits
exactly raw, but historical K-dependent ties may select different candidate IDs.
User explicitly accepted this; NO slow native K fallback is added to OPD.
OPD tree confidence pruning also resolves exact ties by ascending full node
index. FP32 path products can equal their parent's score (conditional p=1 or
underflow to zero); native arbitrary confidence Top-K can otherwise orphan a
child. Exact integer lexicographic keys preserve every confidence bit, select
the same nodes for unique scores, and prefer ancestors on ties. A fused GPU
encoder writes into a reused small tree-key buffer, not a full-vocabulary
sort. This tie rule is OPD-only; historical pruning is untouched, and the
parent-closure device assertion remains enabled. No epsilon/probability change,
additional target forward, RNG draw or host synchronization is introduced.
Proposal q/compact-ID outputs are transient views of reusable scratch. The root
beam/history confidence is copied ONCE into a separate preallocated B*K buffer
before expansion; child confidence multiplication/gathers already create owned
tensors. This prevents later proposals from rewriting earlier probability
products or mixing batch/context rows. No full-vocab cloning. Masked all--inf
logit tiles contribute zero mass; Top16 merge keeps selection eligibility separate
from score so zero-probability slots have unique, valid IDs. Finite-logit
normalization and historical FastGRPO remain unchanged.
Target sampling and verifier rules unchanged. Actual trajectories/round counts
may differ; no extra forwards means existing prefill/verify/expand/committed work
per round, not forcing equal totals when AAL changes.

Sparse prepares ONLY S active corrected scalars. Dense supports fused rank-8
correction + normalization + Top16, or tiled ordered FP32 rank GEMM into reused
workspace. Both use ordered FP32 multiply/add (FMA/TF32 disabled). The fused
implementation avoids a dense workspace write/read. Fixed pairwise tile sums
prevent compiler layout choices from changing the sum association when switching
backends. This can change a few rounding bits versus the previous version's
generic reduction, but not the mathematical distribution. No historical-state
probability cache or extra transformer forward is introduced.

OPD_PROPOSAL_MODE=auto is default (adaptive remains an alias). Exactly ONE backend
is launched per proposal. A GPU active-count snapshot is piggybacked on the ONE
existing host scheduling packet, not read with an extra .item(). Dispatch uses
the previous proposal's active count (one feedback update behind), while BOTH
backends use CURRENT B/bitmap/active IDs for exact correction. The delay can only
affect speed. It preserves side-stream overlap rather than waiting for feedback
early. Active rows are append-only until rollout reset, including canceled rows.

Measured profiles must match GPU/Torch/Triton/CUDA/kernel hash. Thresholds are
interpolated by log(total contexts), with endpoint clamping, so previously unseen
batch factorizations do not fall back to sparse. Without a profile, auto switches
at V/8 active rows: an explicit uncalibrated policy, NOT a measured B200 crossover.
OPD_DENSE_IMPLEMENTATION=auto chooses the measured fused/GEMM implementation in
the nearest context bucket; without measurements it uses fused. Override with
fused/gemm for experiments. Tune on B200 and validate end-to-end.
Counters opd_proposal_mode_sparse_rounds/dense_rounds count ROOT verification
trees, not expansion calls; cold zero-correction trees count sparse/raw.

Runtime scalar extents/strides are no longer constexpr and use do_not_specialize;
fixed power-of-two tile sizes may still have bounded variants. Tree masks tile
columns in blocks of 256, independent of past length. Teacher selection compacts
visited/frontier IDs on GPU. Persistent scan CTAs iterate only selected IDs;
unselected rows do not read target vocab or compute Top16. Already-sampled target
probabilities are reused, without a second target sort/softmax. Compact-vocab
extraction still requires one selected-row scan.

OPD history uses contiguous [original response, capacity, ...] pools. Append
scatters only the newly accepted chunk using GPU owner IDs. Finish records lengths,
not row views or clones. Rare growth for verification padding cannot retain old
storage; one end-of-rollout gather per field materializes compact output storage.
Historical FastGRPO continues using its original per-response history.

OPD-only scheduler pads into fixed capacity, computes path lengths/EOS/prefix
extension on GPU, and transfers ONE small packet at the existing scheduling
boundary. GPU boolean padding masks/position cumulative sums replace per-round
Python padding sets/walks. Accepted real width is preserved, not replaced by
fixed-width fake EOS rows. Exact active batch size and native sampler RNG
consumption retained. One host boundary cannot be removed with DynamicCache.crop
and native dynamic batch sampling without changing those semantics.

KV: only accepted non-prefix suffix gathered into reused per-head small scratch;
write back into current storage, crop views. Accepted prefix not copied. No stack
all KV layers/full-history concat after verification. HF's next native update
creates normal contiguous KV before attention, no untested attention-layout change.
Finished batch index_select still copies remaining KV rows; further static-cache
refactor needs native-model parity and is NOT silently enabled. Historical
baseline keeps original KV work regardless of OPD improvement.

Two dependency events per rollout, side-stream update overlaps commit/KV/next
hidden work, wait only before next proposal. Profile OFF no synchronization.
Profile ON separates feature/correction/select/teacher/union/update; inclusive
proposal and overlapping wait not double-counted. End-to-end wall authoritative.

Metric definitions unchanged: exact cumulative differences per GRPO label,
AAL=sum accepted lengths / SEQUENCE verification rounds, includes target bonus,
not prefill token. KL weighted mean, other coverage/union means per valid selected
state, active rows post-update mean per OPD batch round. Nonfinite KL reported
null with counter; no gradient gate. Same CSV schema both methods.
