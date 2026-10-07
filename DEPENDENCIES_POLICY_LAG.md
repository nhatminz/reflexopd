# Pinned dependency contract

- FastGRPO: `yedaotian9/FastGRPO@38e252493149072d2c5905f0a47de1d935d7170a`
- SpecForge: `sgl-project/SpecForge@3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83` (`0.2.0`)
- Python: project interpreter gate `>=3.12.0`, including `3.12.3`; prefer `3.12.x`
  for the reproduced environment (`>=3.11` is required by the vendored SpecForge
  metadata). Newer interpreters still need compatible pinned dependencies/CUDA.
- PyTorch: `2.13.0` with the official CUDA 13.0 wheel.
- Transformers: `5.12.1`.
- SGLang: `0.5.18`. SGLang is used only for offline SpecForge feature capture,
  not for FastGRPO rollout or verification.

SpecForge source is bundled in `third_party/SpecForge`, including a
`VENDORED_COMMIT` provenance file, so the experiment does not clone or fetch
source code on the B200 machine. `requirements.txt` exactly pins every direct
dependency and `requirements-policy-lag.txt` delegates to it. CUDA framework
binaries must be provided through the offline wheelhouse described in
`ENVIRONMENT.md`. The analysis
code imports SpecForge EAGLE-3's `AutoDraftModel`, `OnlineEagle3Model`, feature
layer rule, model forward, compact full-vocabulary teacher projection, and
training-time TTT objective. It never substitutes the local legacy
SmoothL1/soft-target draft objective for an EAGLE-3 branch.

The development workstation does not have this exact production environment,
so full B200 execution was not launched there. Dependency validation
intentionally fails on a mismatched stack instead of silently selecting another
SpecForge commit or objective.

The launcher imports the concrete EAGLE-3, FlexAttention, and offline SGLang
capture APIs before allocating the target model and rejects versions that do
not match the pinned stack. It writes the actual versions to each pretrain
run's `dependencies.json`.

For targets with `tie_word_embeddings=true`, the frozen SpecForge target head
uses the checkpoint's configured embedding tensor when `lm_head.weight` is
deduplicated from the weight index. This is weight tying, not random or
reinitialized target supervision.
