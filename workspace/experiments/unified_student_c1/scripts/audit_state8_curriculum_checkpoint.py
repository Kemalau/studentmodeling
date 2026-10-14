#!/usr/bin/env python3
"""Audit immutable State8 curriculum checkpoints before read-only screens."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

try:
    from materialize_state8_curriculum_checkpoint_screen import ARMS, STEPS
except ImportError:  # package-style unit tests
    from scripts.materialize_state8_curriculum_checkpoint_screen import ARMS, STEPS


LEGACY_GENERIC_ROUTE = {
    "entrypoint_sha256": "0f4e305fa5eb48ec2beaa328a3b9c125b649f28327a405a211a1803570498a1a",
    "state8_natural_view_sha256": "98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682",
    "decoder_view": "state_plus_current_query_strict_late_fusion",
    "final_inference_route": "strict_state_only",
    "curriculum_route": "qualified_frozen_teacher_to_state8_student",
    "external_state_teacher_scope": "qualified_full_history_answer_logits_only",
    "external_state_teacher_trainable": False,
    "teacher_trainable": False,
    "direct_replay_weight": 0.0,
    "dual_path_curriculum": True,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def raw_history_route_proof(config: dict[str, Any], arm: str) -> str:
    """Prove the decoder route without treating an absent field as false.

    The first generic diagnosis-first curriculum serialized the executable
    route and its immutable entrypoint/view hashes but not the redundant
    ``raw_history_visible_to_decoder`` boolean.  Accept only that exact legacy
    recipe.  Current-query checkpoints and all other recipes must carry the
    explicit field, so a genuinely ambiguous checkpoint still fails closed.
    """
    if "raw_history_visible_to_decoder" in config:
        if config["raw_history_visible_to_decoder"] is not False:
            raise ValueError("Checkpoint config drift: raw history is visible")
        return "explicit_false"
    if arm != "generic":
        raise ValueError("Checkpoint config drift: raw-history route is absent")
    drift = {
        key: config.get(key)
        for key, expected in LEGACY_GENERIC_ROUTE.items()
        if config.get(key) != expected
    }
    if drift:
        raise ValueError(f"Checkpoint legacy route proof failed: {drift}")
    return "legacy_exact_hashed_state_only_route"


def audit(
    root: Path,
    arm: str,
    step: int,
    adapter: str,
    state: str,
    receipt_sha: str,
) -> dict[str, Any]:
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Unsupported curriculum arm/checkpoint step")
    expected_root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        f"{ARMS[arm]}/checkpoint-{step}"
    )
    if str(root) != expected_root:
        raise ValueError(f"Checkpoint root drift: {root}")
    receipt_path = root / "checkpoint_completion.json"
    if digest(receipt_path) != receipt_sha:
        raise ValueError("Checkpoint receipt SHA256 drift")
    expected = {"adapter_model": adapter, "recurrent_state": state}
    observed = {
        "adapter_model": digest(root / "adapter/adapter_model.safetensors"),
        "recurrent_state": digest(root / "recurrent_state.pt"),
    }
    if observed != expected:
        raise ValueError("Checkpoint artifact SHA256 drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    position = receipt.get("step", receipt.get("round"))
    if (
        receipt.get("status") != "complete"
        or int(position or -1) != step
        or receipt.get("sha256") != expected
    ):
        raise ValueError("Checkpoint completion receipt drift")
    payload = torch.load(
        root / "recurrent_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = payload.get("config") or {}
    history = list(payload.get("history") or [])
    payload_position = payload.get("step", payload.get("round"))
    if int(payload_position or -1) != step or len(history) != step:
        raise ValueError("Checkpoint history length drift")
    history_route_proof = raw_history_route_proof(config, arm)
    required = {
        "state_architecture": "layerwise",
        "state_token_count": 8,
        "final_inference_route": "strict_state_only",
        "diagnosis_field_order": ["diagnosis", "evidence", "label"],
        "canonical_label_role": "auxiliary_ordinary_weight",
        "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
        "semantic_judge_authoritative": True,
        "exact_label_metric_authoritative": False,
    }
    for key, expected_value in required.items():
        if config.get(key) != expected_value:
            raise ValueError(f"Checkpoint config drift: {key}={config.get(key)!r}")
    if arm == "current_query":
        if config.get("current_query_routing") != "current_item_visible_response_state_only_v1:eedi_only":
            raise ValueError("Current-query routing drift")
        if config.get("student_response_visible_to_decoder") is not False:
            raise ValueError("Student response leaked to decoder")
    positions = [int(row.get("global_position", -1)) for row in history]
    if positions != list(range(step)):
        raise ValueError("Checkpoint training history is not contiguous")
    counts = Counter(float(row.get("history_dropout", -1)) for row in history)
    expected_counts = {
        level: min(1000, max(0, step - index * 1000))
        for index, level in enumerate((0.0, 0.25, 0.5, 0.75, 1.0))
    }
    expected_counts = Counter({key: value for key, value in expected_counts.items() if value})
    if counts != expected_counts:
        raise ValueError(f"Curriculum exposure drift: {counts} != {expected_counts}")
    return {
        "stage": "state8_curriculum_checkpoint_audit_v1",
        "status": "passed",
        "arm": arm,
        "step": step,
        "state_shape": [64, 8, 5120],
        "history_dropout_counts": dict(sorted(counts.items())),
        "raw_history_visible_to_decoder": False,
        "raw_history_route_proof": history_route_proof,
        "legacy_route_binding": (
            LEGACY_GENERIC_ROUTE if history_route_proof.startswith("legacy_") else None
        ),
        "artifact_sha256": observed,
        "receipt_sha256": receipt_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.checkpoint_root,
        args.arm,
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
