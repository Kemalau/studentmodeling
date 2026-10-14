#!/usr/bin/env python3
"""Select a frozen-decoder layerwise trajectory curriculum by causal/task gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_BRANCHES = {
    "r64_scale": {"transition_rank": 64, "relative_mse_weight": 1.0},
    "r256_scale": {"transition_rank": 256, "relative_mse_weight": 1.0},
    "r256_cosine": {"transition_rank": 256, "relative_mse_weight": 0.0},
}

FINAL_FIT_POLICY = {
    "learning_rate": 5e-7,
    "causal_weight": 1.0,
    "causal_objective_mode": "specificity_triplet",
    "causal_repeat_fraction": 0.05,
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def unique(job_root: Path, name: str) -> Path:
    paths = sorted(job_root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"Expected one {name} under {job_root}, found {paths}")
    return paths[0]


def candidate(job_root: Path) -> dict[str, Any]:
    receipt = read(unique(job_root, "trajectory_curriculum_receipt.json"))
    gate = read(unique(job_root, "recurrent_causal_gate.json"))
    validation = read(unique(job_root, "validation_metrics.json"))
    gradient = read(unique(job_root, "gradient_audit.json"))
    completion = read(unique(job_root, "completion.json"))
    branch = str(receipt.get("branch") or "")
    expected = EXPECTED_BRANCHES.get(branch)
    remote_run_dir = str(receipt.get("remote_run_dir") or "")
    expected_run_dir = (
        "/mnt/amulet/studentmodeling/checkpoints/"
        f"layerwise_trajectory_curriculum_{branch}_seed42_n200_20260731"
    )
    summary = gate["summary"]
    clean = summary["clean"]
    recurrent_gap = float(validation.get("accuracy_gap_recurrent_minus_direct", -1.0))
    transition_gradient = gradient.get("layerwise_transition") or {}
    lora_gradient = gradient.get("lora") or {}
    checks = {
        "training_complete": receipt.get("status") == "trajectory_curriculum_complete"
        and completion.get("status") == "diagnostic_complete",
        "exposure_complete": int(completion.get("micro_steps", -1)) == 200
        and int(completion.get("causal_pairs_seen", -1)) == 200
        and int(receipt.get("optimizer_steps", -1)) == 50,
        "registered_branch": expected is not None
        and int(receipt.get("transition_rank", -1)) == int(expected["transition_rank"])
        and float(receipt.get("relative_mse_weight", -1.0))
        == float(expected["relative_mse_weight"]),
        "closed_decoder": receipt.get("raw_history_visible_to_decoder") is False
        and receipt.get("persistent_history_kv_cache") is False
        and gate.get("raw_history_visible_to_decoder") is False
        and gate.get("persistent_history_kv_cache") is False,
        "remote_curriculum_artifacts": remote_run_dir == expected_run_dir,
        "transition_gradient": float(transition_gradient.get("gradient_abs_sum", 0.0)) > 0,
        "decoder_frozen": int(lora_gradient.get("parameter_tensors", -1)) == 0
        and not gradient.get("frozen_tensors_with_grad"),
        "candidate_support": int(summary.get("clean_n", 0)) >= 50,
        "specificity": float(summary.get("D_specific", -1.0)) >= 0.15,
        "noop": float(clean["noop"]["flip_to_A_rate"]) <= 0.05,
        "same_rule": float(clean["samemal"]["flip_to_A_rate"]) <= 0.10,
        "task_gap": recurrent_gap >= -0.05,
    }
    return {
        "branch": branch,
        "artifact_root": str(job_root),
        "remote_run_dir": remote_run_dir,
        "receipt": receipt,
        "C4": {
            "clean_n": int(summary["clean_n"]),
            "swap": float(clean["swap"]["flip_to_A_rate"]),
            "random": float(clean["random"]["flip_to_A_rate"]),
            "same_rule": float(clean["samemal"]["flip_to_A_rate"]),
            "noop": float(clean["noop"]["flip_to_A_rate"]),
            "D_specific": float(summary["D_specific"]),
            "D_damage": float(summary["D_damage"]),
        },
        "validation": validation,
        "final_fit_policy": dict(FINAL_FIT_POLICY),
        "checks": checks,
        "pass": all(checks.values()),
    }


def select(root: Path) -> dict[str, Any]:
    receipts = sorted(root.rglob("metrics/trajectory_curriculum_receipt.json"))
    job_roots = sorted({path.parents[1] for path in receipts})
    if len(job_roots) != 3:
        raise ValueError(f"Expected three trajectory jobs, found {job_roots}")
    rows = [candidate(path) for path in job_roots]
    if {row["branch"] for row in rows} != set(EXPECTED_BRANCHES):
        raise ValueError(f"Trajectory branch set drift: {[row['branch'] for row in rows]}")
    passing = [row for row in rows if row["pass"]]
    if not passing:
        return {
            "stage": "layerwise_trajectory_curriculum_selection",
            "scientific_gate_pass": False,
            "selected_branch": None,
            "candidates": rows,
        }
    passing.sort(
        key=lambda row: (
            row["C4"]["D_specific"],
            row["C4"]["swap"],
            -row["C4"]["random"],
            -row["receipt"]["transition_rank"],
        ),
        reverse=True,
    )
    selected = passing[0]
    return {
        "stage": "layerwise_trajectory_curriculum_selection",
        "scientific_gate_pass": True,
        "selected_branch": selected["branch"],
        "selected_artifact_root": selected["artifact_root"],
        "selected_C4": selected["C4"],
        "selected_final_fit_policy": selected["final_fit_policy"],
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = select(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["scientific_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
