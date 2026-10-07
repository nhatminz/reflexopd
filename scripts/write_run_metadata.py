#!/usr/bin/env python3
"""Write human-readable resolved run configuration and completion summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--kind", required=True, choices=("pretrain", "train"))
    parser.add_argument("--status", default="configured")
    parser.add_argument("--item", action="append", default=[])
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    values = {"kind": args.kind, "status": args.status}
    for item in args.item:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"--item expects KEY=VALUE, got {item!r}")
        values[key] = value
    yaml_lines = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in values.items()]
    path = run_dir / "config_resolved.yaml"
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    if args.status != "configured":
        summary = run_dir / "summary.json"
        payload = dict(values)
        temporary_summary = summary.with_suffix(".json.tmp")
        temporary_summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary_summary, summary)
        (run_dir / "summary.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in payload.items()) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
