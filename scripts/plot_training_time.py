#!/usr/bin/env python3
"""Plot low-overhead timing logs from one or more SpecNaacl runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", default="training_time.png")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fields = (
        ("cumulative_wall_time_s", "Cumulative wall time (s)"),
        ("cumulative_generation_time_s", "Cumulative generation time (s)"),
        ("tokens_per_s", "Rollout tokens / wall second"),
    )
    for run_value in args.run_dirs:
        run_dir = Path(run_value).expanduser().resolve()
        path = run_dir / "logs" / "timing.csv"
        if not path.is_file():
            raise FileNotFoundError(f"timing log not found: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        steps = [int(row["step"]) for row in rows]
        for axis, (field, label) in zip(axes, fields):
            axis.plot(steps, [float(row[field]) for row in rows], label=run_dir.name)
            axis.set_xlabel("Training step")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    print(output.resolve())


if __name__ == "__main__":
    main()
