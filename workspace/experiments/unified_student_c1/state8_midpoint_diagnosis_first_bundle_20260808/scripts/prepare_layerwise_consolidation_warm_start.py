#!/usr/bin/env python3
"""Derive a weight-identical phase-2 warm start with canonical schedule metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import torch

from audit_layerwise_diagnostic_warm_start import audit


EXPECTED_ROWS = 259_845


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_order(seed: int = 42) -> list[int]:
    order = list(range(EXPECTED_ROWS))
    random.Random(seed).shuffle(order)
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-round", type=int, default=3250)
    parser.add_argument("--expected-world-size", type=int, default=4)
    args = parser.parse_args()

    prior_sha256 = sha256(args.prior_manifest)
    source_audit = audit(
        args.source,
        expected_round=args.expected_round,
        expected_world_size=args.expected_world_size,
        expected_manifest_sha256=prior_sha256,
    )
    prior = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    prior_indices = [int(value) for value in prior["indices"]]
    prior_start = int(prior["start_global_position"])
    prior_end = prior_start + len(prior_indices)
    if prior_start < 0 or prior_end != args.expected_round * args.expected_world_size:
        raise ValueError("Prior curriculum boundary drift")

    source_state = args.source / "recurrent_state.pt"
    payload = torch.load(source_state, map_location="cpu", weights_only=False)
    observed = list(payload.get("real_order") or [])
    expected = canonical_order()
    if len(observed) != len(expected):
        raise ValueError("Warm-start real-order support drift")
    if observed[:prior_start] != expected[:prior_start]:
        raise ValueError("Warm-start order drift before phase-1")
    if observed[prior_start:prior_end] != prior_indices:
        raise ValueError("Warm-start phase-1 order does not match its manifest")
    if observed[prior_end:] != expected[prior_end:]:
        raise ValueError("Warm-start order drift after phase-1")

    if args.output.exists():
        shutil.rmtree(args.output)
    (args.output / "adapter").mkdir(parents=True)
    for source in (args.source / "adapter").iterdir():
        target = args.output / "adapter" / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)

    payload["real_order"] = expected
    config = dict(payload.get("config") or {})
    config.update(
        consolidation_warm_start_derived=True,
        consolidation_source_checkpoint=str(args.source),
        consolidation_prior_manifest=str(args.prior_manifest),
        consolidation_prior_segment_start=prior_start,
        consolidation_prior_segment_end=prior_end - 1,
        consolidation_schedule_metadata_canonicalized=True,
        consolidation_model_weights_modified=False,
    )
    payload["config"] = config
    temporary = args.output / "recurrent_state.pt.partial"
    torch.save(payload, temporary)
    os.replace(temporary, args.output / "recurrent_state.pt")
    hashes = {
        "adapter_model": sha256(args.output / "adapter/adapter_model.safetensors"),
        "recurrent_state": sha256(args.output / "recurrent_state.pt"),
    }
    receipt = {
        "status": "complete",
        "round": args.expected_round,
        "world_size": args.expected_world_size,
        "sha256": hashes,
        "derived_for": "open_t3_causal_task_consolidation",
        "source_sha256": source_audit["sha256"],
        "model_weights_modified": False,
        "schedule_metadata_canonicalized": True,
        "prior_manifest": str(args.prior_manifest),
        "prior_manifest_sha256": prior_sha256,
        "prior_segment_start": prior_start,
        "prior_segment_end": prior_end - 1,
    }
    (args.output / "checkpoint_completion.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
