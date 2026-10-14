#!/usr/bin/env python3
"""Verify the complete 259845-row final-fit artifact and its exposure audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def unique(root: Path, suffix: str) -> Path:
    matches = sorted(root.rglob(suffix))
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix}, found {matches}")
    return matches[0]


def validate(root: Path, arm: str, projection: str) -> dict[str, Any]:
    if arm not in {"generation_only", "generation_kd"}:
        raise ValueError(f"Unknown arm={arm}")
    if projection not in {"identity", "trainable"}:
        raise ValueError(f"Unknown projection={projection}")
    completion_path = unique(root, "metrics/completion.json")
    metrics_root = completion_path.parent
    job_root = metrics_root.parent
    model_root = job_root / "model"
    required = {
        "adapter_model": model_root / "adapter_model.safetensors",
        "recurrent_state": model_root / "recurrent_state.pt",
        "validation_metrics": metrics_root / "validation_metrics.json",
        "causal_validation_metrics": metrics_root / "causal_validation_metrics.json",
        "run_config": metrics_root / "run_config.json",
        "gradient_audit": metrics_root / "gradient_audit.json",
        "training_metrics": metrics_root / "training_metrics.json",
        "data_audit": metrics_root / "data_audit.json",
        "final_fit_manifest": metrics_root / "final_fit_manifest.json",
        "pilot_receipt": metrics_root / "pilot_receipt.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing final-fit artifacts: {missing}")
    completion = read(completion_path)
    config = read(required["run_config"])
    manifest = read(required["final_fit_manifest"])
    gradients = read(required["gradient_audit"])
    pilot = read(required["pilot_receipt"])
    allowed_causal_policies = {
        (1.0, False),
        (4.0, False),
        (1.0, True),
        (4.0, True),
        (8.0, False),
        (2.0, True),
    }
    pilot_policies = {
        None: (1.0, False, 0.05),
        "raw4": (4.0, False, 1726.0 / 12000.0),
        "mean_normalized": (1.0, True, 0.05),
        "raw8": (8.0, False, 1726.0 / 12000.0),
        "mean2_normalized": (2.0, True, 0.05),
        "repeat100_w1": (1.0, True, 0.05),
        "repeat100_w4": (4.0, True, 0.05),
        "repeat100_sum_w1": (1.0, True, 0.05),
        "repeat100_sum_w4": (4.0, True, 0.05),
        "bootstrap_sum_w1": (1.0, True, 0.05),
        "bootstrap_sum_w4": (4.0, True, 0.05),
        "bootstrap_native25_sum_w1": (1.0, True, 0.05),
        "bootstrap_native25_sum_w4": (4.0, True, 0.05),
        "bootstrap_balanced_sum_w1": (1.0, True, 0.05),
        "bootstrap_balanced_sum_w4": (4.0, True, 0.05),
        "bootstrap_native25_balanced_sum_w1": (1.0, True, 0.05),
        "bootstrap_native25_balanced_sum_w4": (4.0, True, 0.05),
        "donor_directional_sum_w1": (1.0, True, 0.05),
        "donor_directional_sum_w4": (4.0, True, 0.05),
        "cf4_bootstrap_symmetric_w1": (1.0, True, 0.05),
        "cf4_donor_directional_w1": (1.0, True, 0.05),
    }
    pilot_branch = pilot.get("causal_rescue_branch")
    expected_pilot_policy = pilot_policies.get(pilot_branch)
    expected_causal_data_mode = pilot.get("causal_data_mode", "cf1_single_query")
    expected_causal_hashes = {
        "cf1_single_query": (
            "edb1bda6cc6aceb4507ac599377284c533014cef34b80e1063e062a74752267e",
            "3e6b5fbcab3d232d5b3a65a005fbf9ba5719a8a4f8251d6ea85b861a7c1c3e22",
        ),
        "cf4_balanced": (
            "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766",
            "3e6b5fbcab3d232d5b3a65a005fbf9ba5719a8a4f8251d6ea85b861a7c1c3e22",
        ),
    }
    expected_causal_hash_pair = expected_causal_hashes.get(expected_causal_data_mode)
    expected_objective_mode = pilot.get(
        "final_fit_causal_objective_mode",
        pilot.get("causal_objective_mode", "symmetric"),
    )
    expected_hashes = completion.get("sha256") or {}
    checks = {
        "complete": completion.get("status") == "complete"
        and completion.get("diagnostic") is False,
        "exact_micro_steps": int(completion.get("micro_steps", -1)) == 259845,
        "exact_real_examples": int(completion.get("real_examples_seen", -1)) == 259845,
        "real_once": completion.get("real_examples_seen_once") is True,
        "causal_count": int(completion.get("causal_pairs_seen", -1))
        == int(config.get("planned_causal_exposures", -2)),
        "causal_count_matches_plan": completion.get(
            "causal_exposure_count_matches_plan"
        )
        is True,
        "all_causal_pairs_covered": completion.get("all_causal_pairs_covered") is True,
        "distributed_exact_global_batch_16": config.get("distributed") is True
        and int(config.get("distributed_world_size", -1))
        * int(config.get("gradient_accumulation_steps", -1))
        == 16,
        "exact_optimizer_steps": int(config.get("planned_optimizer_steps", -1))
        == 16241
        and int(completion.get("optimizer_steps", -1)) == 16241,
        "full_rows_config": int(config.get("real_train_rows", -1)) == 259845
        and int(config.get("full_micro_steps", -1)) == 259845,
        "causal_fraction": 0.0 < float(config.get("causal_repeat_fraction", -1.0)) <= 1.0,
        "causal_policy_inherited": (
            float(config.get("causal_weight", -1.0)),
            bool(config.get("normalize_causal_exposure")),
        )
        in allowed_causal_policies,
        "causal_objective_inherited": config.get("causal_objective_mode")
        == expected_objective_mode,
        "causal_data_mode_registered": expected_causal_hash_pair is not None,
        "causal_data_provenance_inherited": expected_causal_hash_pair is not None
        and (
            config.get("causal_train_sha256"),
            config.get("causal_valid_sha256"),
        )
        == expected_causal_hash_pair,
        "pilot_receipt_status": pilot.get("status") == "candidate_complete"
        and pilot.get("arm") == arm
        and pilot.get("projection_mode") == projection
        and pilot.get("mask_protocol") == "answer_anchor_visible_v2",
        "pilot_artifact_inherited": str(config.get("init_run_dir"))
        == str(pilot.get("run_dir"))
        and str(config.get("effective_init_adapter"))
        == str(Path(str(pilot.get("run_dir"))) / "final_adapter"),
        "pilot_causal_policy_inherited": expected_pilot_policy is not None
        and (
            float(config.get("causal_weight", -1.0)),
            bool(config.get("normalize_causal_exposure")),
            float(config.get("causal_repeat_fraction", -1.0)),
        )
        == (
            float(pilot.get("final_fit_causal_weight", expected_pilot_policy[0])),
            bool(
                pilot.get(
                    "final_fit_normalize_causal_exposure",
                    expected_pilot_policy[1],
                )
            ),
            float(
                pilot.get(
                    "final_fit_causal_repeat_fraction", expected_pilot_policy[2]
                )
            ),
        ),
        "pilot_rescue_C4_gate": pilot_branch is None
        or float(pilot.get("C4_D_specific", -1.0)) >= 0.15,
        "causal_preference_reduction_inherited": config.get(
            "causal_preference_reduction", "mean"
        )
        == pilot.get(
            "final_fit_causal_preference_reduction",
            pilot.get("causal_preference_reduction", "mean"),
        ),
        "closed_decoder": config.get("decoder_view")
        == "state_plus_current_query_strict_late_fusion",
        "exact_exposure_protocol": config.get("real_data_exposure")
        == "exactly_one_epoch_without_replacement",
        "projection_matches": bool(config.get("trainable_projections"))
        == (projection == "trainable"),
        "kd_matches": (float(config.get("behavior_kd_weight", 0.0)) > 0)
        == (arm == "generation_kd"),
        "manifest_rows": int(manifest.get("unique_examples", -1)) == 259845
        and int((manifest.get("counts") or {}).get("rows", -1)) == 259845,
        "manifest_purpose": manifest.get("purpose")
        == "post_selection_final_fit_train_plus_validation",
        "no_frozen_grad": not gradients.get("frozen_tensors_with_grad"),
    }
    for key in (
        "adapter_model",
        "recurrent_state",
        "validation_metrics",
        "causal_validation_metrics",
        "run_config",
        "gradient_audit",
    ):
        checks[f"hash_{key}"] = expected_hashes.get(key) == sha256(required[key])
    payload = {
        "stage": "final_fit_259845",
        "arm": arm,
        "projection_mode": projection,
        "pass": all(checks.values()),
        "checks": checks,
        "completion": completion,
        "run_config": config,
        "artifact_root": str(job_root),
        "artifact_sha256": {
            "adapter_model": sha256(required["adapter_model"]),
            "recurrent_state": sha256(required["recurrent_state"]),
        },
    }
    if not payload["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Final fit failed checks: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root, args.arm, args.projection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
