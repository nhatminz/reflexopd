#!/usr/bin/env python3
"""Read-only checkout integrity check before launching GPU GRPO training."""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def validate_training_sources(repo_root: Path, backend: str):
    if backend not in {"eagle3", "legacy"}:
        raise ValueError("draft backend must be eagle3 or legacy")
    required = [
        "__init__.py", "rewards.py", "get_QAs.py", "specualtive_generate.py",
        "checkpointing.py", "method_config.py", "drift_metrics.py", "rollout_history.py", "step_metrics.py",
        "opd_reflex.py", "opd_reflex_kernels.py", "tree_kernels.py", "sampling.py", "tree_verification.py",
        "historical_fastgrpo.py", "opd_scheduling.py", "opd_static_cache.py",
        "opd_kv_kernels.py", "opd_attention.py", "opd_attention_kernels.py",
        "opd_sampling.py", "opd_history.py", "rollout_metrics.py",
        "opd_profiles.py",
        "shared_rollout.py", "eagle3_online_objective.py",
    ]
    required.append("eagle3_specforge.py" if backend == "eagle3" else "modeling_draft.py")
    # The entrypoint still imports the adapter definition for both backends;
    # its external SpecForge imports are lazy until EAGLE initialization.
    if "eagle3_specforge.py" not in required:
        required.append("eagle3_specforge.py")
    paths = [Path(repo_root) / "helper" / name for name in required]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete SpecNaacl checkout; missing local helper source files:\n- "
            + "\n- ".join(missing)
            + "\nSync the helper/ directory from the same project revision. "
            "These are project sources, not a pip package named 'helper'. "
            "No training or checkpoint change was performed."
        )
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("eagle3", "legacy"))
    args = parser.parse_args(argv)
    try:
        validate_training_sources(REPO_ROOT, args.backend)
    except FileNotFoundError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    print(f"Training source check passed: backend={args.backend} helper={REPO_ROOT / 'helper'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
