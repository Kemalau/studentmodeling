#!/usr/bin/env python3
"""Fail-closed audit for diagnosis-level task-anchor rescue milestones."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT_PREFIX = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_stateonly_repair_seed2067_taskanchor_"
    "matched_cp2000_causal_taskanchor_v1/checkpoint-"
)
STEPS = {200, 400, 600, 800, 1000}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit(
    root: Path,
    step: int,
    adapter_sha256: str,
    state_sha256: str,
    receipt_sha256: str,
) -> dict:
    if step not in STEPS or str(root) != f"{ROOT_PREFIX}{step}":
        raise ValueError("Task-anchor milestone root/step drift")
    expected = {
        "adapter_model": adapter_sha256,
        "recurrent_state": state_sha256,
    }
    if any(
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
        for value in (*expected.values(), receipt_sha256)
    ):
        raise ValueError("Invalid SHA256 binding")
    receipt_path = root / "checkpoint_completion.json"
    if digest(receipt_path) != receipt_sha256:
        raise ValueError("Checkpoint receipt SHA256 drift")
    observed = {
        "adapter_model": digest(root / "adapter/adapter_model.safetensors"),
        "recurrent_state": digest(root / "recurrent_state.pt"),
    }
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if observed != expected or receipt != {
        "sha256": expected,
        "status": "complete",
        "step": step,
    }:
        raise ValueError("Checkpoint artifact binding drift")

    payload = torch.load(
        root / "recurrent_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
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
        "task_anchor_schedule": [1.0, 0.5, 0.25],
        "task_anchor_scope": "diagnose_misconception",
        "task_anchor_visible_at_inference": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Checkpoint config drift: {key}")
    numeric = {
        "label_auxiliary_weight": 1.0,
        "online_kd_weight": 1.0,
        "causal_weight": 4.0,
        "causal_margin": 0.75,
        "causal_repeat_fraction": 1.0,
    }
    for key, value in numeric.items():
        if float(config.get(key, -1)) != value:
            raise ValueError(f"Checkpoint numeric config drift: {key}")
    if config.get("raw_history_visible_to_decoder", False) is not False:
        raise ValueError("Raw history leaked to decoder")
    if not all(
        bool(row.get("history_dropped"))
        and float(row.get("history_dropout", -1)) == 1.0
        and bool(row.get("diagnosis_first_task_anchor_composed"))
        for row in history
    ):
        raise ValueError("Strict state-only/task-anchor history drift")
    if [int(row.get("global_position", -1)) for row in history] != list(range(step)):
        raise ValueError("Training history is not contiguous")
    causal = [row for row in history if row.get("causal_pair_id") is not None]
    diagnosis_causal = [
        row for row in causal if str(row.get("causal_pair_id")).endswith(":diagnosis")
    ]
    answer_causal = [row for row in causal if not str(row.get("causal_pair_id")).endswith(":diagnosis")]
    if not causal or not diagnosis_causal or not answer_causal:
        raise ValueError("Answer/diagnosis causal exposure missing")
    return {
        "stage": "state8_taskanchor_milestone_audit_v1",
        "status": "passed",
        "step": step,
        "optimizer_steps": (step + 3) // 4,
        "state_shape": [64, 8, 5120],
        "history_dropout_counts": {"1.0": step},
        "causal_records": len(causal),
        "answer_causal_records": len(answer_causal),
        "diagnosis_causal_records": len(diagnosis_causal),
        "raw_history_visible_to_decoder": False,
        "artifact_sha256": observed,
        "receipt_sha256": receipt_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.checkpoint_root,
        args.step,
        args.adapter_sha256,
        args.state_sha256,
        args.receipt_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
