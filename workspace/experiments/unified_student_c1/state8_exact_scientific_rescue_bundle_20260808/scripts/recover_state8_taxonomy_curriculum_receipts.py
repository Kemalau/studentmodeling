#!/usr/bin/env python3
"""Reconstruct missing State8 curriculum receipts from immutable complete artifacts.

This recovery path is intentionally metadata-only.  It never changes model weights,
optimizer state, predictions, or measured metrics.  It fails closed unless the
published completion hashes, exact one-epoch history, frozen teacher provenance,
five-stage curriculum, taxonomy contract, and strict-state-only route all agree.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from state8_natural_curriculum_view import apply_state8_natural_curriculum_view
from state8_taxonomy_contract import apply_state8_taxonomy_contract


EXPECTED_DROPOUT_COUNTS = {0.0: 1000, 0.25: 1000, 0.5: 1000, 0.75: 1000, 1.0: 1000}
EXPECTED_TEACHER_EVAL_SHA256 = "78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5"
EXPECTED_TEACHER_PROMPT_SHA256 = "c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314"
EXPECTED_NATURAL_VIEW_SHA256 = "98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    if len(history) != 5000:
        raise AssertionError(f"Expected 5000 history rows, got {len(history)}")
    positions = [int(row["global_position"]) for row in history]
    if positions != list(range(5000)):
        raise AssertionError("Training history is not the exact contiguous 0..4999 epoch")
    example_ids = [str(row["example_id"]) for row in history]
    if len(set(example_ids)) != 5000:
        raise AssertionError("Training history contains duplicate example IDs")
    dropout_counts = collections.Counter(float(row["history_dropout"]) for row in history)
    if dict(dropout_counts) != EXPECTED_DROPOUT_COUNTS:
        raise AssertionError(f"History-dropout curriculum drift: {dropout_counts}")
    causal_rows = [row for row in history if row.get("causal_pair_id") is not None]
    causal_pairs = {str(row["causal_pair_id"]) for row in causal_rows}
    if len(causal_rows) != 250 or len(causal_pairs) != 250:
        raise AssertionError(
            f"Causal exposure drift: rows={len(causal_rows)} pairs={len(causal_pairs)}"
        )
    if any(str(row.get("causal_objective_phase")) != "specificity_triplet" for row in causal_rows):
        raise AssertionError("Causal objective is not the specificity triplet")
    task_counts = collections.Counter(str(row["task"]) for row in history)
    return {
        "history_rows": len(history),
        "history_unique_examples": len(set(example_ids)),
        "history_dropout_counts": dict(sorted(dropout_counts.items())),
        "causal_rows": len(causal_rows),
        "unique_causal_pairs": len(causal_pairs),
        "task_counts": dict(sorted(task_counts.items())),
    }


def audit_dataset(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    natural = apply_state8_natural_curriculum_view(rows)
    taxonomy = apply_state8_taxonomy_contract(rows)
    if int(natural.get("changed_t3_rows", 0)) <= 0:
        raise AssertionError(f"Dataset contains no natural T3 rows: {path}")
    if int(taxonomy.get("taxonomy_contract_rows", 0)) <= 0:
        raise AssertionError(f"Dataset contains no taxonomy contracts: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "audit_provenance": "reconstructed_from_immutable_complete_artifacts",
        **natural,
        **taxonomy,
    }


def require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AssertionError(f"{label} drift: observed={observed!r} expected={expected!r}")


def recover(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_dir.resolve()
    required = {
        "completion": root / "completion.json",
        "config": root / "run_config.json",
        "history": root / "trainer_history.json",
        "training_metrics": root / "training_metrics.json",
        "validation_metrics": root / "validation_metrics.json",
        "causal_validation_metrics": root / "causal_validation_metrics.json",
        "gradient_audit": root / "gradient_audit.json",
        "adapter": root / "final_adapter/adapter_model.safetensors",
        "recurrent_state": root / "recurrent_state.pt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Complete-artifact recovery inputs are missing: {missing}")

    completion = read_json(required["completion"])
    config = read_json(required["config"])
    require_equal(completion.get("status"), "complete", "completion status")
    require_equal(completion.get("diagnostic"), False, "diagnostic flag")
    for key, expected in {
        "micro_steps": 5000,
        "optimizer_steps": 1250,
        "real_examples_seen": 5000,
        "history_rows": 5000,
        "history_unique_global_positions": 5000,
        "history_unique_examples": 5000,
        "causal_pairs_seen": 250,
        "unique_causal_pairs_seen": 250,
        "real_examples_seen_once": True,
        "all_causal_pairs_covered": True,
        "causal_exposure_count_matches_plan": True,
        "exact_history_support": True,
        "history_integrity_pass": True,
    }.items():
        require_equal(completion.get(key), expected, f"completion.{key}")

    hashes = completion.get("sha256") or {}
    hash_inputs = {
        "adapter_model": required["adapter"],
        "recurrent_state": required["recurrent_state"],
        "validation_metrics": required["validation_metrics"],
        "causal_validation_metrics": required["causal_validation_metrics"],
        "run_config": required["config"],
        "gradient_audit": required["gradient_audit"],
    }
    observed_hashes = {key: file_sha256(path) for key, path in hash_inputs.items()}
    require_equal(observed_hashes, hashes, "completion artifact hashes")

    expected_config = {
        "state_token_count": 8,
        "state_architecture": "layerwise",
        "layerwise_transition_rank": 256,
        "history_dropout_curriculum": True,
        "dual_path_curriculum": True,
        "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
        "online_kd_weight": 1.0,
        "online_kd_temperature": 1.0,
        "online_kd_topk": 256,
        "consistency_weight": 0.0,
        "direct_replay_weight": 0.0,
        "causal_weight": 2.0,
        "causal_margin": 0.5,
        "causal_objective_mode": "specificity_triplet",
        "normalize_causal_exposure": True,
        "state8_natural_view_sha256": EXPECTED_NATURAL_VIEW_SHA256,
        "teacher_prompt_provenance_stage": "state8_clean_prompt_teacher_provenance_gate_v2",
        "teacher_evaluation_input_sha256": EXPECTED_TEACHER_EVAL_SHA256,
        "teacher_clean_prompt_sha256": EXPECTED_TEACHER_PROMPT_SHA256,
        "legacy_answer_adjacent_contracts": 0,
        "teacher_gate_required": True,
        "teacher_trainable": False,
        "external_state_teacher_trainable": False,
        "trajectory_teacher_scope": "none_unverified_carrier_trajectory_excluded",
        "final_inference_route": "strict_state_only",
        "decoder_view": "state_plus_current_query_strict_late_fusion",
        "output_schema": "label_diagnosis_evidence_v1",
        "label_auxiliary_weight": 4.0,
        "taxonomy_label_contract": "source_aware_no_answer_labels_v1",
        "taxonomy_candidates_visible_to_decoder": False,
        "semantic_judge_authoritative": True,
        "exact_label_metric_authoritative": False,
        "real_train_rows": 5000,
        "planned_optimizer_steps": 1250,
        "planned_causal_exposures": 250,
        "distributed_world_size": 4,
    }
    for key, expected in expected_config.items():
        require_equal(config.get(key), expected, f"run_config.{key}")
    require_equal(file_sha256(args.train_jsonl), config.get("train_sha256"), "train data hash")
    require_equal(file_sha256(args.valid_jsonl), config.get("valid_sha256"), "valid data hash")

    history_value = json.loads(required["history"].read_text(encoding="utf-8"))
    if not isinstance(history_value, list):
        raise TypeError("trainer_history.json must contain a list")
    history_audit = audit_history(history_value)

    gradient = read_json(required["gradient_audit"])
    if gradient.get("frozen_tensors_with_grad") != []:
        raise AssertionError("Frozen tensors received gradients")
    for group in ("lora", "layerwise_transition"):
        if float((gradient.get(group) or {}).get("gradient_abs_sum", 0.0)) <= 0.0:
            raise AssertionError(f"Missing nonzero gradient for {group}")

    teacher_model = args.teacher_root / "model/adapter_model.safetensors"
    teacher_gate = read_json(args.teacher_root / "metrics/teacher_gate.json")
    teacher_provenance = read_json(args.teacher_root / "metrics/teacher_provenance_gate.json")
    teacher_publish = read_json(args.teacher_root / "metrics/publish_receipt.json")
    teacher_hash = file_sha256(teacher_model)
    require_equal(teacher_hash, config.get("external_state_teacher_sha256"), "teacher adapter hash")
    require_equal(teacher_gate.get("pass"), True, "teacher gate")
    require_equal(teacher_gate.get("support"), 1494, "teacher gate support")
    require_equal(teacher_gate.get("exact_match_is_authoritative"), False, "teacher exact-match role")
    require_equal(teacher_provenance.get("status"), "complete", "teacher provenance status")
    require_equal(teacher_provenance.get("teacher_gate_pass"), True, "teacher provenance gate")
    require_equal(teacher_provenance.get("evaluation_input_sha256"), EXPECTED_TEACHER_EVAL_SHA256, "teacher eval hash")
    require_equal(teacher_provenance.get("clean_prompt_sha256"), EXPECTED_TEACHER_PROMPT_SHA256, "teacher prompt hash")
    require_equal(teacher_provenance.get("legacy_answer_adjacent_contracts"), 0, "teacher prompt leakage")
    require_equal(teacher_publish.get("status"), "complete", "teacher publish status")
    require_equal(teacher_publish.get("adapter_sha256"), teacher_hash, "teacher publish hash")

    dataset_audits = [audit_dataset(args.train_jsonl), audit_dataset(args.valid_jsonl)]
    external_receipt = {
        "stage": "state8_qualified_external_teacher_curriculum_v1",
        "teacher_adapter": str((args.teacher_root / "model").resolve()),
        "teacher_adapter_sha256": teacher_hash,
        "teacher_gate_required": True,
        "teacher_trainable": False,
        "state_token_count": 8,
        "output_schema": "label_diagnosis_evidence_v1",
        "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
        "online_teacher_scope": config["online_teacher_scope"],
        "trajectory_teacher_scope": "none_unverified_carrier_trajectory_excluded",
        "final_inference_route": "strict_state_only",
        "state8_natural_view_sha256": EXPECTED_NATURAL_VIEW_SHA256,
        "teacher_prompt_provenance_stage": "state8_clean_prompt_teacher_provenance_gate_v2",
        "teacher_evaluation_input_sha256": EXPECTED_TEACHER_EVAL_SHA256,
        "teacher_clean_prompt_sha256": EXPECTED_TEACHER_PROMPT_SHA256,
        "legacy_answer_adjacent_contracts": 0,
        "dataset_audits": dataset_audits,
        "receipt_recovery": "immutable_complete_artifacts_only",
        "source_completion_sha256": file_sha256(required["completion"]),
    }
    taxonomy_receipt = {
        "stage": "state8_taxonomy_teacher_curriculum_v1",
        "status": "complete",
        "state_token_count": 8,
        "output_schema": "label_diagnosis_evidence_v1",
        "history_dropout_schedule": [0.0, 0.25, 0.5, 0.75, 1.0],
        "label_auxiliary_weight": 4.0,
        "taxonomy_contract": "source_aware_no_answer_labels_v1",
        "taxonomy_candidates_visible_to_decoder": False,
        "semantic_judge_authoritative": True,
        "exact_label_metric_authoritative": False,
        "final_inference_route": "strict_state_only",
        "raw_history_visible_to_decoder": False,
        "audits": dataset_audits,
        "receipt_recovery": "immutable_complete_artifacts_only",
        "source_completion_sha256": file_sha256(required["completion"]),
    }
    qualified_receipt = {
        "status": "complete",
        "stage": "state8_qualified_teacher_curriculum_v1",
        "replay_variant": "t3_heavy",
        "micro_steps": 5000,
        "state_shape": [64, 8, 5120],
        "history_dropout_counts": history_audit["history_dropout_counts"],
        "online_kd_weight": 1.0,
        "direct_offload_min_tokens": 1,
        "teacher_adapter_sha256": teacher_hash,
        "raw_history_visible_to_decoder": False,
        "teacher_prompt_provenance_stage": "state8_clean_prompt_teacher_provenance_gate_v2",
        "teacher_evaluation_input_sha256": EXPECTED_TEACHER_EVAL_SHA256,
        "teacher_clean_prompt_sha256": EXPECTED_TEACHER_PROMPT_SHA256,
        "legacy_answer_adjacent_contracts": 0,
        "label_auxiliary_weight": 4.0,
        "taxonomy_label_contract": "source_aware_no_answer_labels_v1",
        "semantic_judge_authoritative": True,
        "exact_label_metric_authoritative": False,
        "persistent_history_kv_cache": False,
        "adapter_sha256": observed_hashes["adapter_model"],
        "recurrent_state_sha256": observed_hashes["recurrent_state"],
        "receipt_recovery": "immutable_complete_artifacts_only",
        "source_completion_sha256": file_sha256(required["completion"]),
    }
    atomic_json(root / "state8_external_teacher_curriculum.json", external_receipt)
    atomic_json(root / "taxonomy_teacher_curriculum.json", taxonomy_receipt)
    atomic_json(root / "qualified_teacher_receipt.json", qualified_receipt)
    recovery = {
        "stage": "state8_taxonomy_curriculum_receipt_recovery_v1",
        "status": "complete",
        "model_or_metric_mutation": False,
        "run_dir": str(root),
        "source_artifact_sha256": observed_hashes,
        "history_audit": history_audit,
        "dataset_audits": dataset_audits,
        "teacher_adapter_sha256": teacher_hash,
        "written_receipts": {
            name: file_sha256(root / name)
            for name in (
                "state8_external_teacher_curriculum.json",
                "taxonomy_teacher_curriculum.json",
                "qualified_teacher_receipt.json",
            )
        },
    }
    atomic_json(root / "receipt_recovery.json", recovery)
    return recovery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--valid-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    args = parser.parse_args()
    result = recover(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
