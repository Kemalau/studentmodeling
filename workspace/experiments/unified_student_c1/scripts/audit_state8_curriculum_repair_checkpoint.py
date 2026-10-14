#!/usr/bin/env python3
"""Audit immutable State8 curriculum-repair checkpoints before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SPECS: dict[tuple[int, str], dict[str, Any]] = {
    (5201, "semantic_binding"): {"source_step": 4000},
    (5202, "specificity_triplet"): {"source_step": 4000},
    (5203, "balanced_continuation"): {"source_step": 4000},
    (6201, "semantic_binding"): {"source_step": 3000},
    (6202, "specificity_triplet"): {"source_step": 3000},
    (6203, "balanced_continuation"): {"source_step": 3000},
    (8201, "causal_teacher_low"): {
        "source_step": 3000,
        "causal_teacher_trajectory_weight": 0.15,
    },
    (8202, "causal_teacher_high"): {
        "source_step": 3000,
        "causal_teacher_trajectory_weight": 0.50,
    },
    (8301, "early_label_low"): {
        "source_step": 3000,
        "causal_teacher_trajectory_weight": 0.15,
        "early_label_binding_weight": 0.25,
    },
    (8302, "early_label_high"): {
        "source_step": 3000,
        "causal_teacher_trajectory_weight": 0.15,
        "early_label_binding_weight": 0.50,
    },
}
STEPS = {20, 100, 240, 500}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit(
    root: Path,
    seed: int,
    branch: str,
    step: int,
    adapter: str,
    state: str,
    receipt_sha: str,
) -> dict[str, Any]:
    spec = SPECS.get((seed, branch))
    if spec is None or step not in STEPS:
        raise ValueError("Unsupported curriculum repair checkpoint")
    expected_root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        "state8_curriculum_repairs_20260812/"
        f"seed{seed}_{branch}/checkpoint-{step}"
    )
    if str(root) != expected_root:
        raise ValueError(f"Repair checkpoint root drift: {root}")
    receipt_path = root / "checkpoint_completion.json"
    if digest(receipt_path) != receipt_sha:
        raise ValueError("Repair checkpoint receipt SHA256 drift")
    expected = {"adapter_model": adapter, "recurrent_state": state}
    observed = {
        "adapter_model": digest(root / "adapter/adapter_model.safetensors"),
        "recurrent_state": digest(root / "recurrent_state.pt"),
    }
    if observed != expected:
        raise ValueError("Repair checkpoint artifact SHA256 drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != {"sha256": expected, "status": "complete", "step": step}:
        raise ValueError("Repair checkpoint completion receipt drift")
    payload = torch.load(
        root / "recurrent_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = payload.get("config") or {}
    history = list(payload.get("history") or [])
    if int(payload.get("step", -1)) != step or len(history) != step:
        raise ValueError("Repair checkpoint history length drift")
    required = {
        "state_architecture": "layerwise",
        "state_token_count": 8,
        "final_inference_route": "strict_state_only",
        "diagnosis_field_order": ["diagnosis", "evidence", "label"],
        "canonical_label_role": "auxiliary_ordinary_weight",
        "history_dropout_schedule": [1.0],
        "raw_history_visible_to_decoder": False,
        "current_query_routing": "current_item_visible_response_state_only_v1:eedi_only",
        "current_item_visible_to_decoder": True,
        "student_response_visible_to_decoder": False,
        "source_anchor_train_only": True,
        "source_visible_at_inference": False,
        "semantic_random_target": "insufficient_evidence",
        "semantic_auxiliary_visible_at_inference": False,
    }
    for key, expected_value in required.items():
        if config.get(key) != expected_value:
            raise ValueError(
                f"Repair checkpoint config drift: {key}={config.get(key)!r}"
            )
    if [int(row.get("global_position", -1)) for row in history] != list(range(step)):
        raise ValueError("Repair checkpoint history is not contiguous")
    if not all(
        float(row.get("history_dropout", -1)) == 1.0
        and bool(row.get("history_dropped"))
        and bool(row.get("source_anchor_train_only"))
        for row in history
    ):
        raise ValueError("Repair checkpoint strict-state history drift")
    teacher_weight = spec.get("causal_teacher_trajectory_weight")
    if teacher_weight is not None:
        if float(config.get("causal_teacher_trajectory_weight", -1)) != float(
            teacher_weight
        ):
            raise ValueError("Causal teacher-binding weight drift")
        if config.get("causal_teacher_trajectory_scope") != (
            "native_and_donor_diagnosis_pairs_train_only"
        ):
            raise ValueError("Causal teacher-binding scope drift")
        causal_diagnosis = [
            row
            for row in history
            if row.get("causal_pair_id") is not None
            and str(row.get("causal_teacher_binding_source") or "")
            in {"eedi_mining", "malrule"}
        ]
        if not causal_diagnosis:
            raise ValueError("Causal teacher binding has no checkpoint support")
        expected_effective_weight = float(teacher_weight) * float(
            config.get("causal_exposure_multiplier", 1.0)
        )
        if any(
            float(row.get("causal_teacher_binding_effective_weight", -1))
            != expected_effective_weight
            or float(row.get("causal_teacher_binding_loss", -1)) < 0
            for row in causal_diagnosis
        ):
            raise ValueError("Causal teacher-binding history drift")
    early_label_weight = spec.get("early_label_binding_weight")
    early_label_support = 0
    if early_label_weight is not None:
        expected_config = {
            "early_label_binding_weight": float(early_label_weight),
            "early_label_binding_scope": "diagnosis_causal_pairs_train_only",
            "early_label_binding_visible_at_inference": False,
            "early_label_binding_metric_role": (
                "auxiliary_supervision_not_authoritative_evaluation"
            ),
        }
        for key, expected_value in expected_config.items():
            if config.get(key) != expected_value:
                raise ValueError(f"Early-label config drift: {key}")
        early_rows = [
            row
            for row in history
            if float(row.get("early_label_binding_effective_weight", 0.0)) > 0.0
        ]
        if not early_rows:
            raise ValueError("Early-label binding has no checkpoint support")
        expected_effective_weight = float(early_label_weight) * float(
            config.get("causal_exposure_multiplier", 1.0)
        )
        if any(
            float(row.get("early_label_binding_effective_weight", -1))
            != expected_effective_weight
            or float(row.get("early_label_binding_loss", -1)) < 0
            or row.get("early_label_target_mode")
            != "identifier_only_no_gold_prefix"
            for row in early_rows
        ):
            raise ValueError("Early-label binding history drift")
        early_label_support = len(early_rows)
    return {
        "stage": "state8_curriculum_repair_checkpoint_audit_v1",
        "status": "passed",
        "seed": seed,
        "branch": branch,
        "source_checkpoint_step": spec["source_step"],
        "step": step,
        "state_shape": [64, 8, 5120],
        "raw_history_visible_to_decoder": False,
        "artifact_sha256": observed,
        "receipt_sha256": receipt_sha,
        "causal_teacher_trajectory_weight": teacher_weight,
        "causal_teacher_binding_support": (
            len(causal_diagnosis) if teacher_weight is not None else 0
        ),
        "early_label_binding_weight": early_label_weight,
        "early_label_binding_support": early_label_support,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--state-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.checkpoint_root,
        args.seed,
        args.branch,
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
