#!/usr/bin/env python3
"""Completion audit for the promoted exact-epoch layerwise unified model."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from validate_layerwise_unified_pipeline import validate as validate_base


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {name}: {matches}")
    return matches[0]


def link_tree(source: Path, destination: Path, *, skip: set[str] | None = None) -> None:
    skip = skip or set()
    for path in source.rglob("*"):
        if not path.is_file() or path.name in skip:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(path.resolve(), target)


def validate(root: Path) -> dict[str, Any]:
    fit_dir = root / "task_retention_promoted_final_fit"
    eval_dir = root / "task_retention_promoted_final_eval"
    semantic_dir = root / "task_retention_promoted_t3_semantic_judge"
    markers = (
        fit_dir / "complete",
        fit_dir / "scientific_gate_pass",
        eval_dir / "complete",
        eval_dir / "scientific_gate_pass",
        semantic_dir / "complete",
        semantic_dir / "scientific_gate_pass",
    )
    missing = [str(path) for path in markers if not path.is_file()]
    if missing:
        raise ValueError(f"Missing promoted completion markers: {missing}")

    receipt_path = one(fit_dir / "download", "promoted_final_fit_receipt.json")
    receipt = read(receipt_path)
    config = read(one(fit_dir / "download", "run_config.json"))
    promotion = read(one(fit_dir / "download", "promoted_real_order_manifest.json"))
    source_audit = read(one(fit_dir / "download", "source_checkpoint_audit.json"))
    promoted_checks = {
        "receipt": (
            receipt.get("status") == "final_fit_complete"
            and receipt.get("stage") == "layerwise_task_retention_promoted_final_fit"
            and receipt.get("branch")
            in {"equal_t3_t6", "equal_t3_t6_open_t3", "equal_t3_t6_open_t3_two_stage", "t3_heavy", "eedi_dense", "staged_t3_t6", "eedi_full", "eedi_full_open_t3"}
            and receipt.get("loss_profile") in {"standard", "state_kd", "state_path"}
            and receipt.get("candidate_id")
            == (
                f"{receipt.get('branch')}__{receipt.get('loss_profile')}__lr{float(receipt.get('learning_rate')):.0e}"
                if receipt.get("branch") in {"eedi_full", "eedi_full_open_t3"}
                or float(receipt.get("learning_rate")) != 5e-7
                else f"{receipt.get('branch')}__{receipt.get('loss_profile')}"
            )
            and receipt.get("promoted_source_round") == 750
            and receipt.get("one_epoch_exact") is True
            and float(receipt.get("causal_weight", -1.0)) == 2.0
        ),
        "order": (
            promotion.get("stage") == "layerwise_task_retention_promoted_order_v1"
            and promotion.get("branch") == receipt.get("branch")
            and promotion.get("selected_loss_profile") == receipt.get("loss_profile")
            and promotion.get("selected_candidate_id") == receipt.get("candidate_id")
            and promotion.get("source_checkpoint_round") == 750
            and promotion.get("source_world_size") == 16
            and promotion.get("completed_prefix_rows") == 12000
            and promotion.get("epoch_rows") == 259845
            and promotion.get("promoted_rows")
            == (
                5000
                if receipt.get("branch") in {"eedi_full", "eedi_full_open_t3"}
                else 2000
                if receipt.get("branch") == "equal_t3_t6_open_t3_two_stage"
                else 1000
            )
            and float(promotion.get("selected_learning_rate", -1))
            == float(receipt.get("learning_rate", -2))
            and float(promotion.get("selected_causal_weight", -1))
            == float(receipt.get("causal_weight", -2)) == 2.0
            and promotion.get("one_epoch_permutation") is True
            and promotion.get("completed_prefix_unchanged") is True
            and promotion.get("formal_example_id_overlap") == 0
            and promotion.get("real_order_sha256") == receipt.get("promoted_order_sha256")
        ),
        "config": (
            config.get("promoted_real_order_applied") is True
            and config.get("promoted_real_order_source_round") == 750
            and config.get("promoted_real_order_manifest_sha256")
            and float(config.get("promoted_selected_causal_weight", -1.0)) == 2.0
            and float(config.get("causal_weight", -1.0)) == 2.0
        ),
        "source_checkpoint": (
            source_audit.get("status") == "passed"
            and source_audit.get("exact_resume_ready") is True
            and source_audit.get("checkpoint_round") == 750
            and source_audit.get("world_size") == 16
            and source_audit.get("history_rows") == 12000
        ),
    }
    if "t6_decision_threshold" in receipt:
        calibration = read(one(fit_dir / "download", "t6_calibration.json"))
        threshold_audit = read(one(eval_dir / "download", "t6_threshold_application.json"))
        import hashlib

        calibration_path = one(fit_dir / "download", "t6_calibration.json")
        promoted_checks["t6_calibration"] = (
            receipt.get("t6_threshold_frozen_before_final_fit") is True
            and receipt.get("t6_threshold_selection_split")
            == "unused_original_validation"
            and calibration.get("stage")
            == "layerwise_t6_threshold_calibration_v1"
            and calibration.get("status") == "frozen"
            and float(calibration.get("threshold", -1))
            == float(receipt.get("t6_decision_threshold", -2))
            and hashlib.sha256(calibration_path.read_bytes()).hexdigest()
            == receipt.get("t6_calibration_artifact_sha256")
            and threshold_audit.get("stage")
            == "layerwise_t6_frozen_threshold_application_v1"
            and float(threshold_audit.get("threshold", -1))
            == float(receipt.get("t6_decision_threshold", -2))
            and threshold_audit.get("calibration_artifact_sha256")
            == receipt.get("t6_calibration_artifact_sha256")
            and threshold_audit.get("rows_updated") == 976
        )
    if receipt.get("branch") in {"eedi_full_open_t3", "equal_t3_t6_open_t3", "equal_t3_t6_open_t3_two_stage"}:
        if receipt.get("loss_profile") == "state_kd" or receipt.get("branch") == "equal_t3_t6_open_t3_two_stage":
            open_view = read(
                one(
                    fit_dir / "download",
                    "promoted_open_t3_selective_kd_training_view.json",
                )
            )
            promoted_checks["open_t3_training_view"] = (
                open_view.get("stage")
                == "promoted_open_t3_selective_kd_training_view_v1"
                and open_view.get("scope") == "unseen_exact_epoch_suffix_only"
                and open_view.get("global_behavior_kd_weight") == 0.5
                and open_view.get("excluded_behavior_kd_weight") == 0.0
                and open_view.get("direct_replay_weight") == 0.0
                and int(open_view.get("excluded_history_rows", 0)) > 0
                and int(open_view.get("retained_history_rows", 0)) > 0
                and open_view.get("causal_rows_modified") is False
                and open_view.get("validation_rows_modified") is False
            )
        else:
            open_view = read(
                one(fit_dir / "download", "promoted_open_t3_training_view.json")
            )
            promoted_checks["open_t3_training_view"] = (
                open_view.get("stage") == "promoted_open_t3_training_view_v1"
                and open_view.get("scope") == "unseen_exact_epoch_suffix_only"
                and open_view.get("behavior_kd_weight") == 0.0
                and open_view.get("direct_replay_weight") == 0.0
                and open_view.get("causal_rows_modified") is False
                and open_view.get("validation_rows_modified") is False
                and int(open_view.get("changed_malrule_t3_rows", 0)) > 0
            )
    if receipt.get("branch") == "equal_t3_t6_open_t3_two_stage":
        transition = read(one(fit_dir / "download", "two_stage_transition_receipt.json"))
        promoted_checks["two_stage_curriculum"] = (
            receipt.get("two_stage_curriculum") is True
            and promotion.get("curriculum_mode")
            == "open_t3_formation_then_causal_task_consolidation"
            and promotion.get("formation_transition_global_position") == 13008
            and config.get("promoted_two_stage_phase") == "consolidation"
            and config.get("promoted_two_stage_transition_applied") is True
            and transition.get("stage") == "promoted_two_stage_transition_v1"
            and transition.get("global_position") == 13008
            and transition.get("optimizer_moments_preserved") is True
        )
    if not all(promoted_checks.values()):
        raise ValueError(
            f"Promoted checks failed: {[key for key, value in promoted_checks.items() if not value]}"
        )

    # Reuse the stricter base audit for exact epoch support, data provenance,
    # causal exposure, task/C4 gates, T3 judgment and artifact hashes.  Only a
    # small compatibility receipt is synthesized; large artifacts stay linked.
    with tempfile.TemporaryDirectory(prefix="promoted-unified-audit-") as temporary:
        compat = Path(temporary)
        compat_fit = compat / "final_fit"
        compat_eval = compat / "final_eval"
        compat_semantic = compat / "t3_semantic_judge"
        for directory in (compat_fit, compat_eval, compat_semantic):
            directory.mkdir(parents=True)
            (directory / "complete").touch()
            (directory / "scientific_gate_pass").touch()
        link_tree(
            fit_dir / "download",
            compat_fit / "download",
            skip={"promoted_final_fit_receipt.json"},
        )
        compatible_receipt = dict(receipt)
        compatible_receipt["stage"] = "layerwise_unified_final_fit"
        receipt_target = compat_fit / "download/compat/metrics/final_fit_receipt.json"
        receipt_target.parent.mkdir(parents=True, exist_ok=True)
        receipt_target.write_text(json.dumps(compatible_receipt), encoding="utf-8")
        os.symlink((fit_dir / "artifact_manifest.json").resolve(), compat_fit / "artifact_manifest.json")
        os.symlink((eval_dir / "summary.json").resolve(), compat_eval / "summary.json")
        os.symlink(
            (semantic_dir / "t3_semantic_gate.json").resolve(),
            compat_semantic / "t3_semantic_gate.json",
        )
        base = validate_base(compat)

    return {
        "version": 1,
        "status": "complete",
        "objective": "full_promoted_unified_layerwise_recurrent_student_model",
        "selected_branch": receipt["branch"],
        "selected_loss_profile": receipt["loss_profile"],
        "selected_candidate": receipt["candidate_id"],
        "promoted_checks": promoted_checks,
        "base_checks": base["checks"],
        "artifacts": {
            "adapter_model": str(one(fit_dir / "download", "adapter_model.safetensors")),
            "recurrent_state": str(one(fit_dir / "download", "recurrent_state.pt")),
        },
        "gates": {
            "final_fit": str(fit_dir),
            "formal_and_C4": str(eval_dir / "summary.json"),
            "T3_semantic": str(semantic_dir / "t3_semantic_gate.json"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
