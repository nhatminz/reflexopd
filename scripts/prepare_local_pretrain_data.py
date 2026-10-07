#!/usr/bin/env python3
"""Convert the four documented local datasets to ShareGPT JSON for SpecForge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _text(value):
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, dict):
            return str(first.get("content", first))
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".json", ".jsonl"}:
        if source.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(source).to_dict(orient="records")
    else:
        raise ValueError(f"unsupported pretrain data format: {source.suffix}")
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    converted = []
    for index, row in enumerate(rows):
        if isinstance(row.get("conversations"), list):
            converted.append(row)
            continue
        question = row.get("question", row.get("prompt", row.get("problem")))
        answer = row.get("answer", row.get("solution", row.get("response")))
        reward_model = row.get("reward_model")
        if answer is None and isinstance(reward_model, dict):
            answer = reward_model.get("ground_truth")
        if question is None or answer is None:
            raise ValueError(f"row {index} has no supported prompt/answer fields")
        converted.append({
            "id": str(row.get("id", index)),
            "conversations": [
                {"from": "human", "value": _text(question)},
                {"from": "gpt", "value": _text(answer)},
            ],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(converted, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Converted {len(converted)} rows: {source} -> {output}")


if __name__ == "__main__":
    main()
