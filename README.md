# SpecNaacl: fair shared FastGRPO vs OPD Reflex

Both methods share persistent/growable target/draft KV, minimal-move compaction,
mask/position/tree/path workspaces, contiguous history and scheduling/verifier.
FastGRPO proposal remains historical FP32 raw softmax + native torch.topk(draft_k),
including its confidence TopK tie semantics. No A/B, teacher union, correction
dispatcher or OPD update stream is constructed for FastGRPO. Frozen historical
code remains an external correctness reference, not the default timing baseline.

METHOD=opd_reflex always uses fused Top16, including B=0/cold. It uses the
post-norm/head input, persistent LEARNED rank8 A, one shared rollout-local B.
A gradients accumulate GPU-only and apply at existing draft optimizer boundaries.
B resets each rollout. No extra target/draft transformer forward/backward per round.

Per user decision, Top16[:draft_k] ties may differ from historical K. Cold
corrected logits equal raw logits; probabilities still normalize the same full
compact distribution (FP32 reduction tolerance), no temperature/sampling change.
No slow identity fallback in OPD.

OPD_PROPOSAL_MODE=auto uses compatible measured profiles; correction-specific
sparse/fused/GEMM optimization remains OPD-only. No strategy-selection GPU sync.
OPD's existing tied-candidate/parent-closure policy is retained; it is not imposed
on historical FastGRPO's native TopK.

Shared persistent online EAGLE3 loss: 2*SmoothL1(predicted_next_H, target_final_H)
+ 0.1*soft-label CE, shifted one token, excluding prompt/padding, per-response
valid-generated-position normalization. Teacher probabilities detach; compact
support uses fixed d2t mapping and compact-mass renormalization. Architecture and
checkpoint format retained; SpecForge pretraining unchanged. Online TTT/KL/LK
is no longer the persistent objective. OPD A/B objective remains separate.

Defaults: target LR1e-6, draft LR1e-4, draft accumulation1, OPD fast LR0.01,
OPD update_stream1. All environment/CLI overrides remain. Per-iteration loss
components are in logs/rollout_timing.csv. One scheduling host boundary remains.

Commands: huongdanchay.md. Algorithm/metric details: METHOD_OPD_REFLEX.md.
Validation/performance caveats: IMPLEMENTATION_REPORT.md. No B200/full-model
AAL/throughput improvement claimed without real checkpoint measurements.
