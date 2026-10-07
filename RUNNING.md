# Current running interface

LK Reflex mode/launchers were removed. Only METHOD=fastgrpo/opd_reflex remains.
See [huongdanchay.md](huongdanchay.md) for all paired launchers, exportable knobs,
bounded smoke training and actual B200 frozen-rollout LR/stream sweep.
See [METHOD_OPD_REFLEX.md](METHOD_OPD_REFLEX.md) for exactness/backend caveats
and metric definitions. Existing model/data/outputs/pretrain paths are unchanged.
Generic wrappers: scripts/run_fastgrpo_fair.sh and scripts/run_opd_reflex.sh.
