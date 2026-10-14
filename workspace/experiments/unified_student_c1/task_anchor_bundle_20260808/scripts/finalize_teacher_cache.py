#!/usr/bin/env python3
"""Verify every teacher-cache shard and write an immutable integrity receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.cache_dir
    manifest_path = root / "manifest.json"
    index_path = root / "index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or len(index) != int(manifest["rows"]):
        raise ValueError("Teacher cache is incomplete")
    files = sorted({str(row["file"]) for row in index})
    if len(files) != int(manifest["shards"]):
        raise ValueError("Teacher cache shard count mismatch")
    seen = set()
    shard_hashes = {}
    captured_mass = []
    for filename in files:
        path = root / filename
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"Empty or invalid cache shard: {filename}")
        overlap = seen & payload.keys()
        if overlap:
            raise ValueError(f"Duplicate cache examples across shards: {sorted(overlap)[:3]}")
        seen.update(payload)
        for record in payload.values():
            mass = record["teacher_topk_logprobs"].float().exp().sum(dim=-1)
            captured_mass.append(mass)
        shard_hashes[filename] = sha256(path)
    expected = {str(row["example_id"]) for row in index}
    if seen != expected:
        raise ValueError(
            f"Teacher cache example set mismatch: missing={len(expected-seen)} extra={len(seen-expected)}"
        )
    mass_values = torch.cat(captured_mass) if captured_mass else torch.empty(0)
    if not mass_values.numel():
        raise ValueError("Teacher cache has no supervised target positions")
    receipt = {
        "version": 1,
        "status": "verified",
        "manifest_sha256": sha256(manifest_path),
        "index_sha256": sha256(index_path),
        "rows": len(index),
        "shards": shard_hashes,
        "topk_captured_probability_mass": {
            "positions": int(mass_values.numel()),
            "mean": float(mass_values.mean()),
            "p01": float(torch.quantile(mass_values, 0.01)),
            "minimum": float(mass_values.min()),
            "maximum": float(mass_values.max()),
        },
    }
    temporary = root / "integrity.json.partial"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "integrity.json")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
