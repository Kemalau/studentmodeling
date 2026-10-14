#!/usr/bin/env python3
"""Fail-closed audit for a completed diagnostic checkpoint used as phase-2 weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from select_latest_distributed_checkpoint import sha256


EXPECTED_ROWS = 259_845
EXPECTED_TRAIN_SHA256 = (
    "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd"
)
EXPECTED_VALID_SHA256 = (
    "78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14"
)
EXPECTED_CAUSAL_TRAIN_SHA256 = (
    "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766"
)
EXPECTED_CAUSAL_VALID_SHA256 = (
    "8e038c3a60c2e1b420779cd54eb203b12ae5d51544c87a805fe19d47fdbe294e"
)
DEFAULT_DIAGNOSTIC_MANIFEST_SHA256 = (
    "32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(
    checkpoint: Path,
    *,
    expected_round: int,
    expected_world_size: int,
    expected_manifest_sha256: str = DEFAULT_DIAGNOSTIC_MANIFEST_SHA256,
) -> dict[str, Any]:
    state = checkpoint / "recurrent_state.pt"
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    receipt_path = checkpoint / "checkpoint_completion.json"
    require(state.is_file(), "recurrent state is missing")
    require(adapter.is_file(), "adapter is missing")
    require(receipt_path.is_file(), "checkpoint completion receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("status") == "complete", "checkpoint receipt is incomplete")
    require(int(receipt.get("round", -1)) == expected_round, "receipt round drift")
    require(int(receipt.get("world_size", -1)) == expected_world_size, "receipt world-size drift")
    hashes = {
        "adapter_model": sha256(adapter),
        "recurrent_state": sha256(state),
    }
    require(hashes == receipt.get("sha256"), "checkpoint payload hash drift")

    payload = torch.load(state, map_location="cpu", weights_only=False)
    require(payload.get("distributed") is True, "warm start is not distributed")
    require(int(payload.get("round", -1)) == expected_round, "payload round drift")
    require(int(payload.get("world_size", -1)) == expected_world_size, "payload world-size drift")
    require(isinstance(payload.get("optimizer"), dict), "optimizer state is missing")
    require(isinstance(payload.get("scheduler"), dict), "scheduler state is missing")
    rank_rng = payload.get("rank_rng_states") or []
    require(len(rank_rng) == expected_world_size, "rank RNG support drift")
    real_order = payload.get("real_order") or []
    require(len(real_order) == EXPECTED_ROWS, "real-order support drift")
    # A diagnostic curriculum deliberately replaces one slice of the original
    # epoch permutation with audited task-retention indices.  It can therefore
    # contain duplicates globally; the trainer separately proves that every
    # position outside that inherited slice still matches the seed-42 order.
    schedule = payload.get("causal_schedule") or {}
    require(len(schedule) == 12_992, "causal schedule support drift")
    config = payload.get("config") or {}
    expected = {
        "state_architecture": "layerwise",
        "state_token_alias": "<|vision_pad|>",
        "layerwise_transition_rank": 256,
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "valid_sha256": EXPECTED_VALID_SHA256,
        "causal_train_sha256": EXPECTED_CAUSAL_TRAIN_SHA256,
        "causal_valid_sha256": EXPECTED_CAUSAL_VALID_SHA256,
        "seed": 42,
        "causal_objective_mode": "specificity_triplet",
        "causal_repeat_fraction": 0.05,
        "normalize_causal_exposure": True,
    }
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    require(not drift, "warm-start config drift: " + json.dumps(drift, sort_keys=True))
    prior_start = int(config.get("diagnostic_start_global_position", -1))
    prior_rounds = int(config.get("diagnostic_segment_rounds", -1))
    prior_rows = prior_rounds * expected_world_size
    require(prior_start >= 0 and prior_rows > 0, "prior diagnostic boundary is missing")
    require(prior_start + prior_rows == expected_round * expected_world_size, "prior diagnostic boundary drift")
    require(
        config.get("diagnostic_index_manifest_sha256")
        == expected_manifest_sha256,
        "prior diagnostic manifest drift",
    )
    history = payload.get("history") or []
    require(bool(history), "warm-start training history is empty")
    positions = [int(row["global_position"]) for row in history]
    require(positions == list(range(min(positions), max(positions) + 1)), "history positions are not contiguous")
    require(max(positions) + 1 == expected_round * expected_world_size, "history boundary drift")
    return {
        "stage": "layerwise_diagnostic_warm_start_audit_v1",
        "status": "passed",
        "checkpoint_round": expected_round,
        "world_size": expected_world_size,
        "next_global_position": expected_round * expected_world_size,
        "history_rows": len(history),
        "history_global_position_start": min(positions),
        "history_global_position_end": max(positions),
        "real_order_rows": len(real_order),
        "real_order_unique_rows": len(set(real_order)),
        "prior_diagnostic_start": prior_start,
        "prior_diagnostic_rows": prior_rows,
        "causal_exposures": len(schedule),
        "sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-round", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument(
        "--expected-manifest-sha256",
        default=DEFAULT_DIAGNOSTIC_MANIFEST_SHA256,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(
        args.checkpoint,
        expected_round=args.expected_round,
        expected_world_size=args.expected_world_size,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
