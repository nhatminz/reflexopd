#!/usr/bin/env python3
"""Check EAGLE's chosen backend without capture/training or implicit fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "SpecForge"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("fa", "sdpa", "flex_attention"))
    parser.add_argument("--probe", action="store_true", help="exercise FA CUDA forward/backward")
    args = parser.parse_args(argv)

    from specforge.training.pretrain_attention import validate_attention_backend

    try:
        backend = validate_attention_backend(args.backend, probe=args.probe)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "To run without the external FlashAttention package, add "
            "PRETRAIN_ATTENTION_BACKEND=sdpa to your original pretrain command.\n"
            "Keep the other model/dataset/batch settings unchanged. "
            "No backend or training setting was changed automatically.",
            file=sys.stderr,
        )
        return 2
    # SDPA/flex are explicit configuration choices, not validated CUDA probes.
    if backend == "fa":
        result = "CUDA forward/backward probe passed" if args.probe else "required APIs available"
    else:
        result = "explicit selection; no external FlashAttention API required"
    print(f"Pretraining attention: {backend} ({result})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
