#!/usr/bin/env python3
"""Fail-closed audit for immutable source-anchor training checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ARMS: dict[str, dict[str, Any]] = {
    "full_mse1": {"seed": 3071, "tag": "source_anchor_full_mse1_v1", "relative_mse": 1.0, "transition_only": False},
    "updater_mse1": {"seed": 3072, "tag": "source_anchor_updater_mse1_v1", "relative_mse": 1.0, "transition_only": True},
    "current_query_updater_mse1": {
        "seed": 3072,
        "tag": "source_anchor_updater_mse1_current_query_v1",
        "relative_mse": 1.0,
        "transition_only": True,
        "current_query": True,
    },
    "eedi_same_question_updater_mse1": {
        "seed": 3072,
        "tag": "source_anchor_updater_mse1_eedi_same_question_v1",
        "relative_mse": 1.0,
        "transition_only": True,
        "current_query": "eedi_only",
    },
    "eedi_same_question_full_mse1": {
        "seed": 3071,
        "tag": "source_anchor_full_mse1_eedi_same_question_v1",
        "relative_mse": 1.0,
        "transition_only": False,
        "current_query": "eedi_only",
    },
    "full_mse2": {"seed": 3073, "tag": "source_anchor_full_mse2_v1", "relative_mse": 2.0, "transition_only": False},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit(root: Path, arm: str, step: int, adapter: str, state: str, receipt: str) -> dict[str, Any]:
    if arm not in ARMS or step <= 0 or step % 20:
        raise ValueError("Unsupported source-anchor arm/checkpoint step")
    spec = ARMS[arm]
    expected_root = (
        "/mnt/amulet/studentmodeling/checkpoints/"
        f"state8_stateonly_repair_seed{spec['seed']}_taskanchor_{spec['tag']}/checkpoint-{step}"
    )
    snapshot_root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        f"state8_eedi_same_question_20260811/"
        f"{'updater' if 'updater' in arm else 'full'}/checkpoint-{step}"
        if arm.startswith("eedi_same_question_") else None
    )
    if str(root) not in {expected_root, snapshot_root}:
        raise ValueError(f"Checkpoint root drift: {root}")
    expected = {"adapter_model": adapter, "recurrent_state": state}
    hashes = (*expected.values(), receipt)
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes):
        raise ValueError("Invalid SHA256 binding")
    receipt_path = root / "checkpoint_completion.json"
    if digest(receipt_path) != receipt:
        raise ValueError("Checkpoint receipt SHA256 drift")
    observed = {
        "adapter_model": digest(root / "adapter/adapter_model.safetensors"),
        "recurrent_state": digest(root / "recurrent_state.pt"),
    }
    if observed != expected:
        raise ValueError("Checkpoint artifact SHA256 drift")
    value = json.loads(receipt_path.read_text())
    if value != {"sha256": expected, "status": "complete", "step": step}:
        raise ValueError("Checkpoint completion receipt drift")
    payload = torch.load(root / "recurrent_state.pt", map_location="cpu", weights_only=False, mmap=True)
    config = payload.get("config") or {}
    history = list(payload.get("history") or [])
    if int(payload.get("step", -1)) != step or len(history) != step:
        raise ValueError("Checkpoint history length drift")
    required = {
        "state_architecture": "layerwise",
        "state_token_count": 8,
        "final_inference_route": "strict_state_only",
        "diagnosis_field_order": ["diagnosis", "evidence", "label"],
        "canonical_label_role": "auxiliary_ordinary_weight",
        "history_dropout_schedule": [1.0],
        "raw_history_visible_to_decoder": False,
        "source_anchor_train_only": True,
        "source_visible_at_inference": False,
        "trajectory_cosine_weight": 1.0,
        "trajectory_relative_mse_weight": spec["relative_mse"],
        "layerwise_transition_only": spec["transition_only"],
    }
    if spec.get("current_query"):
        routing = (
            "current_item_visible_response_state_only_v1"
            if spec["current_query"] is True
            else f"current_item_visible_response_state_only_v1:{spec['current_query']}"
        )
        required.update(
            current_query_routing=routing,
            current_item_visible_to_decoder=True,
            student_response_visible_to_decoder=False,
        )
    for key, expected_value in required.items():
        if config.get(key) != expected_value:
            raise ValueError(f"Checkpoint config drift: {key}={config.get(key)!r}")
    if [int(row.get("global_position", -1)) for row in history] != list(range(step)):
        raise ValueError("Checkpoint training history is not contiguous")
    if not all(
        bool(row.get("history_dropped"))
        and float(row.get("history_dropout", -1)) == 1.0
        and bool(row.get("source_anchor_train_only"))
        and float(row.get("source_anchor_relative_mse_weight", -1)) == spec["relative_mse"]
        for row in history
    ):
        raise ValueError("Checkpoint strict-state/source-anchor history drift")
    return {
        "stage": "state8_source_anchor_checkpoint_audit_v1",
        "status": "passed",
        "arm": arm,
        "step": step,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "artifact_sha256": observed,
        "receipt_sha256": receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint_root, args.arm, args.step, args.adapter_sha256, args.state_sha256, args.receipt_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
