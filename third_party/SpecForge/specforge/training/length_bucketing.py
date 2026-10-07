"""Deterministic offline batch plans and a cached, tensor-free length index.

Only sample order changes. The existing normalizer/collator still own all
truncation, padding, attention/loss masks and feature alignment.
"""

from __future__ import annotations

import bisect
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import torch


def bucketed_indices(
    lengths: Sequence[int], *, batch_size: int, dp_rank: int, dp_size: int,
    seed: int, epoch: int, boundaries: Sequence[int],
) -> list[int]:
    """Shuffle within buckets and shuffle global batches before rank slicing.

    The final short global batch stays last. Only the usual DistributedSampler
    padding (at most dp_size-1 repeated refs) is needed for equal rank lengths;
    no source sample is dropped and single-rank plans never duplicate samples.
    """
    if batch_size < 1 or dp_size < 1 or not 0 <= dp_rank < dp_size:
        raise ValueError("invalid batch size or DP layout")
    bounds = list(boundaries)
    if not bounds or bounds != sorted(set(bounds)) or bounds[0] < 1:
        raise ValueError("length bucket boundaries must be positive and strictly increasing")
    if not lengths:
        return []
    if min(lengths) < 1:
        raise ValueError("offline features must have positive sequence lengths")
    generator = torch.Generator().manual_seed(int(seed) + int(epoch))
    indices = torch.randperm(len(lengths), generator=generator).tolist()
    indices.sort(key=lambda index: bisect.bisect_left(bounds, lengths[index]))
    quantum = batch_size * dp_size
    batches = [indices[start:start + quantum] for start in range(0, len(indices), quantum)]
    tail = batches.pop() if len(batches[-1]) < quantum else None
    batch_order = torch.randperm(len(batches), generator=generator).tolist()
    ordered = [batches[index] for index in batch_order]
    if tail is not None:
        padding = (-len(tail)) % dp_size
        if padding:
            tail = tail + (tail * math.ceil(padding / len(tail)))[:padding]
        ordered.append(tail)
    return [index for batch in ordered for index in batch[dp_rank::dp_size]]


def _feature_length(path: str) -> int:
    # Fake tensors read shapes without loading multi-MB hidden-state storages.
    # gzip must decompress for zip seeks, but only on the initial index build.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as stream:
                raw = torch.load(stream, map_location="cpu", weights_only=True)
        else:
            raw = torch.load(path, map_location="cpu", weights_only=True)
    ids = raw["input_ids"]
    if ids.ndim != 1 or ids.shape[0] < 1:
        raise ValueError(f"{path}: expected nonempty 1D offline input_ids, got {ids.shape}")
    return int(ids.shape[0])


def load_length_index(paths: Sequence[str], cache_path: str, max_len: int) -> tuple[list[int], str]:
    """Validate cached lengths against paths/stats; scan only new/changed files."""
    cache = Path(cache_path)
    entries = {}
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("version") == 1:
                entries = saved["entries"]
        except (ValueError, KeyError):
            pass
    result = {}
    lengths = []
    for index, path in enumerate(paths):
        stat = os.stat(path)
        signature = [stat.st_size, stat.st_mtime_ns]
        previous = entries.get(path)
        length = (
            int(previous["length"])
            if previous is not None and previous["stat"] == signature
            else _feature_length(path)
        )
        result[path] = {"stat": signature, "length": length}
        lengths.append(min(length, max_len))
        if (index + 1) % 5000 == 0:
            print(f"Offline length index: {index + 1}/{len(paths)} samples", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"version": 1, "entries": result}), encoding="utf-8")
    os.replace(temporary, cache)
    # The identity also protects a mid-epoch resume against changed features.
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return lengths, digest


def distributed_length_index(paths: Sequence[str], cache_path: str, max_len: int):
    """One filesystem scan, shared with peers; broadcast failures as well."""
    import torch.distributed as dist

    distributed = dist.is_initialized() and dist.get_world_size() > 1
    payload = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            payload[0] = {"result": load_length_index(paths, cache_path, max_len)}
        except Exception as exc:
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
    if distributed:
        dist.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError(f"offline length index failed: {payload[0]['error']}")
    return payload[0]["result"]
