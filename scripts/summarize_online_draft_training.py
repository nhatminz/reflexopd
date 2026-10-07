#!/usr/bin/env python3
"""Compare completed real-training modes; never manufacture speedups."""
import argparse
import json
from pathlib import Path


def summarize(root):
    reports = {}
    for mode in ("per_response", "batched"):
        path = Path(root) / mode / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"incomplete benchmark: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        reports[mode] = {
            "grpo_steps": summary["completed_grpo_steps"],
            "rollout_tokens": summary["total_rollout_tokens"],
            "draft_supervised_tokens": summary["draft_sparse_count"],
            "draft_train_s": summary["total_draft_train_time_s"],
            "wall_s": summary["total_wall_time_s"],
            "generation_tokens_per_s": summary["generation_tokens_per_s"],
            "end_to_end_tokens_per_s": summary["total_rollout_tokens"] / summary["total_wall_time_s"],
            "aal": summary["average_accept_length"],
            "acceptance_rate": summary["draft_acceptance_rate"],
            "peak_allocated_bytes": summary.get("peak_allocated_bytes"),
            "peak_reserved_bytes": summary.get("peak_reserved_bytes"),
        }
    old, new = reports["per_response"], reports["batched"]
    comparable = old["grpo_steps"] == new["grpo_steps"] and old["grpo_steps"] > 0
    return {
        "benchmark": "real_online_draft_training",
        "modes": reports,
        "matched_grpo_steps": comparable,
        "matched_draft_supervised_tokens": old["draft_supervised_tokens"] == new["draft_supervised_tokens"],
        "matched_rollout_tokens": old["rollout_tokens"] == new["rollout_tokens"],
        "draft_training_speedup": old["draft_train_s"] / new["draft_train_s"] if comparable and new["draft_train_s"] > 0 else None,
        "end_to_end_speedup": old["wall_s"] / new["wall_s"] if comparable and new["wall_s"] > 0 else None,
        "note": "Includes load/JIT/startup. Compare matched steps, warmup and GPU memory before choosing the production mode.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
