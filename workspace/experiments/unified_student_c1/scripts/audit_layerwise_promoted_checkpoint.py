#!/usr/bin/env python3
"""Additional fail-closed recipe audit for a promoted distributed checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit(checkpoint: Path, manifest_path: Path, *, expected_round: int) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not an object")
    if payload.get("distributed") is not True or int(payload.get("world_size", -1)) != 16:
        raise ValueError("promoted checkpoint topology drift")
    if int(payload.get("round", -1)) != expected_round:
        raise ValueError("promoted checkpoint round drift")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("promoted checkpoint config is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage") != "layerwise_task_retention_promoted_order_v1":
        raise ValueError("promoted manifest stage drift")
    manifest_sha = file_sha256(manifest_path)
    expected = {
        "promoted_real_order_applied": True,
        "promoted_real_order_manifest_sha256": manifest_sha,
        "promoted_real_order_source_round": 750,
        "promoted_selected_candidate_id": manifest.get("selected_candidate_id"),
        "promoted_selected_loss_profile": manifest.get("selected_loss_profile"),
        "promoted_selected_learning_rate": float(manifest.get("selected_learning_rate", -1.0)),
        "promoted_selected_causal_weight": float(manifest.get("selected_causal_weight", -1.0)),
        "learning_rate": float(manifest.get("selected_learning_rate", -1.0)),
        "causal_weight": float(manifest.get("selected_causal_weight", -1.0)),
    }
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if drift:
        raise ValueError(
            "promoted checkpoint recipe drift: " + json.dumps(drift, sort_keys=True)
        )
    return {
        "stage": "layerwise_promoted_checkpoint_recipe_audit",
        "status": "passed",
        "checkpoint_round": expected_round,
        "world_size": 16,
        "manifest_sha256": manifest_sha,
        "candidate_id": manifest["selected_candidate_id"],
        "loss_profile": manifest["selected_loss_profile"],
        "learning_rate": float(manifest["selected_learning_rate"]),
        "causal_weight": float(manifest["selected_causal_weight"]),
        "source_round": 750,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.checkpoint, args.manifest, expected_round=args.expected_round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
