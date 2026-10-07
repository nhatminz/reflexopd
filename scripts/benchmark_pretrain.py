#!/usr/bin/env python3
"""Short real EAGLE3 offline benchmark. Invoke through benchmark_pretrain.sh.

Uses the production builder, dataset, objective and full-run LR horizon. Only
the stop limit and output directory differ. Timing/synchronization hooks exist
only in this opt-in runner, never in default pretraining.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "SpecForge"))


def validate_benchmark_output(configured_output, requested_output):
    output = Path(configured_output).resolve()
    if output != Path(requested_output).resolve():
        raise ValueError("extra overrides must not redirect the isolated benchmark output")
    # The shell creates this directory and tee may already have opened its log.
    # Never enter an existing run and trigger checkpoint timeline rewinding.
    if output.exists() and any(entry.name != "console.log" for entry in output.iterdir()):
        raise ValueError("benchmark output must be a new/empty directory, not an existing run")


def summarize_benchmark(*, elapsed_s, samples, steps, gpu_count, attention_backend,
                        distributed_mode, warmup_steps, effective_batch):
    if steps < 1 or elapsed_s <= 0:
        raise ValueError("benchmark needs at least one measured optimizer step")
    return {
        "optimizer_step_time_s": elapsed_s / steps,
        "samples_per_second": samples / elapsed_s,
        "samples/s": samples / elapsed_s,
        "gpu_count": gpu_count,
        "attention_backend": attention_backend,
        "distributed_mode": distributed_mode,
        "warmup_optimizer_steps": warmup_steps,
        "measured_optimizer_steps": steps,
        "effective_batch": effective_batch,
        "measured_global_samples": samples,
        "elapsed_s_slowest_rank": elapsed_s,
        "timing_scope": "optimizer boundaries incl. data wait; excludes startup and final checkpoint",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, help="isolated fresh benchmark directory")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("require 0 <= warmup-steps < steps")

    import torch
    import torch.distributed as dist
    from accelerate.utils import set_seed
    from specforge.config import load_config, apply_overrides
    from specforge.application import resolve_run
    from specforge.distributed import init_distributed, destroy_distributed
    from specforge.runtime.data_plane.offline_reader import list_feature_files
    from specforge.training.assembly import build_training_run
    from specforge.training.pretrain_attention import validate_attention_backend

    cfg = load_config(args.config, overrides=args.overrides)
    try:
        validate_benchmark_output(cfg.output_dir, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if cfg.training.strategy != "eagle3" or cfg.mode != "offline":
        parser.error("benchmark is for offline EAGLE3 pretraining only")
    if cfg.training.resume_from:
        parser.error("benchmark uses an isolated new run; do not resume the production run")
    if cfg.profiling.enabled:
        parser.error("disable profiling for a throughput benchmark")
    if cfg.data.max_length != 2048 or cfg.training.ttt_length != 7:
        parser.error("preserve data.max_length=2048 and training.ttt_length=7")
    if (cfg.training.tp_size != 1 or cfg.training.sp_ulysses_size != 1
            or cfg.training.sp_ring_size != 1):
        parser.error("this replicated-data benchmark requires trainer TP/SP=1")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    count = len(list_feature_files(cfg.data.hidden_states_path))
    per_rank = math.ceil(count / world)
    # Match the complete production epoch horizon, not a 30-step cosine run.
    horizon = cfg.training.total_steps or cfg.training.max_steps or (
        math.ceil(per_rank / cfg.training.batch_size) * cfg.training.num_epochs
    ) // cfg.training.accumulation_steps
    if horizon < args.steps:
        parser.error(f"dataset/config provides only {horizon} optimizer steps")
    cfg = apply_overrides(cfg, [
        f"training.max_steps={args.steps}", f"training.total_steps={horizon}",
        "training.save_interval=0", "training.eval_interval=0",
        "tracking.report_to=none",
    ])
    if not torch.cuda.is_available():
        raise RuntimeError("Run this benchmark on CUDA/B200; no CPU speedup claim is valid")
    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    set_seed(cfg.training.seed)
    init_distributed(timeout=cfg.training.dist_timeout, tp_size=1,
                     sp_ulysses_size=1, sp_ring_size=1)
    try:
        validate_attention_backend(cfg.training.attention_backend, probe=True)
        resolved = resolve_run(cfg)
        run = build_training_run(resolved.config, algorithm=resolved.algorithm)
        trainer = run.trainer
        original_step = trainer.core.train_step
        durations, measured_samples = [], []
        window_samples = 0
        optimizer_steps = 0
        torch.cuda.synchronize()
        dist.barrier()
        boundary_time = time.perf_counter()

        def measured_step(batch, ctx=None):
            nonlocal boundary_time, window_samples, optimizer_steps
            window_samples += len(batch.sample_ids)
            result = original_step(batch, ctx)
            if result.optimizer_stepped:
                torch.cuda.synchronize()
                ended = time.perf_counter()
                optimizer_steps += 1
                if optimizer_steps > args.warmup_steps:
                    durations.append(ended - boundary_time)
                    measured_samples.append(window_samples)
                window_samples = 0
                boundary_time = ended
            return result

        trainer.core.train_step = measured_step
        run.run()  # includes the real trainer's normal final checkpoint/resume artifacts
        if optimizer_steps != args.steps:
            raise RuntimeError(f"requested {args.steps} steps, observed {optimizer_steps}")
        elapsed = torch.tensor(sum(durations), device="cuda", dtype=torch.float64)
        samples = torch.tensor(sum(measured_samples), device="cuda", dtype=torch.int64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(samples, op=dist.ReduceOp.SUM)
        report = summarize_benchmark(
            elapsed_s=elapsed.item(), samples=samples.item(), steps=len(durations),
            gpu_count=world, attention_backend=cfg.training.attention_backend,
            distributed_mode=trainer.backend._wrapper_kind.replace("none", "plain"),
            warmup_steps=args.warmup_steps,
            effective_batch=cfg.training.batch_size * cfg.training.accumulation_steps * world,
        )
        report["scheduler_total_steps"] = horizon
        if dist.get_rank() == 0:
            output = Path(cfg.output_dir) / "benchmark_report.json"
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("BENCHMARK:", json.dumps(report, sort_keys=True), flush=True)
            print(f"Report: {output}", flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
