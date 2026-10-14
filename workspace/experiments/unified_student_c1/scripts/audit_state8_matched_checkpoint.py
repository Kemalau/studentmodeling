#!/usr/bin/env python3
"""Fail-closed audit shared by read-only matched State8 checkpoint screens."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


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
    expected_root = (
        "/mnt/amulet/studentmodeling/checkpoints/"
        "state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_"
        "taxonomy_semantic_v1_diagnosis_first_matched_teacher_seed67_single_ga4/"
        f"checkpoint-{step}"
    )
    if str(root) != expected_root:
        raise ValueError(f"Checkpoint root drift: {root}")
    if step not in {2000, 3000, 4000, 5000}:
        raise ValueError("Unsupported checkpoint step")
    adapter = root / "adapter/adapter_model.safetensors"
    state = root / "recurrent_state.pt"
    receipt_path = root / "checkpoint_completion.json"
    if digest(receipt_path) != receipt_sha256:
        raise ValueError("Checkpoint receipt SHA256 drift")
    receipt = json.loads(receipt_path.read_text())
    observed = {"adapter_model": digest(adapter), "recurrent_state": digest(state)}
    expected = {"adapter_model": adapter_sha256, "recurrent_state": state_sha256}
    if observed != expected or receipt != {"sha256": expected, "status": "complete", "step": step}:
        raise ValueError("Checkpoint artifact binding drift")
    payload = torch.load(state, map_location="cpu", weights_only=False, mmap=True)
    config = payload.get("config") or {}; history = payload.get("history") or []
    if int(payload.get("step", -1)) != step or len(history) != step:
        raise ValueError("Checkpoint history length drift")
    required = {
        "state_architecture": "layerwise",
        "state_token_count": 8,
        "final_inference_route": "strict_state_only",
        "diagnosis_field_order": ["diagnosis", "evidence", "label"],
        "canonical_label_role": "auxiliary_ordinary_weight",
        "taxonomy_label_contract": "source_aware_no_answer_labels_v1",
        "teacher_prompt_provenance_stage": "state8_diagnosis_first_teacher_provenance_gate_v1",
    }
    for name, value in required.items():
        if config.get(name) != value:
            raise ValueError(f"Checkpoint config drift: {name}")
    if float(config.get("label_auxiliary_weight", -1)) != 1.0:
        raise ValueError("Canonical label weight drift")
    if config.get("history_dropout_schedule") != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise ValueError("History-dropout curriculum drift")
    if config.get("raw_history_visible_to_decoder", False) is not False:
        raise ValueError("Decoder history leakage")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    counts = {str(value): sum(float(row["history_dropout"]) == value for row in history) for value in levels}
    expected_counts = {str(value): max(0, min(1000, step - index * 1000)) for index, value in enumerate(levels)}
    if counts != expected_counts:
        raise ValueError(f"History-dropout counts drift: {counts}")
    return {
        "stage": "state8_matched_checkpoint_audit_v1",
        "status": "passed",
        "step": step,
        "sha256": observed,
        "receipt_sha256": receipt_sha256,
        "state_shape": [64, 8, 5120],
        "history_dropout_counts": counts,
        "raw_history_visible_to_decoder": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(
        args.checkpoint_root,
        args.step,
        args.adapter_sha256,
        args.state_sha256,
        args.receipt_sha256,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
