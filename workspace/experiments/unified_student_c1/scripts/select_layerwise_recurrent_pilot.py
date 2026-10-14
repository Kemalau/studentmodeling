#!/usr/bin/env python3
"""Select a layerwise recurrent pilot using frozen task and causal gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_BRANCHES = {
    "preserve_low_lr": {
        "learning_rate": 5e-7,
        "causal_weight": 0.5,
        "causal_objective_mode": "donor_directional",
    },
    "balanced": {
        "learning_rate": 2e-6,
        "causal_weight": 1.0,
        "causal_objective_mode": "donor_directional",
    },
    "specificity": {
        "learning_rate": 2e-6,
        "causal_weight": 1.0,
        "causal_objective_mode": "specificity_triplet",
    },
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def candidate(job_root: Path) -> dict[str, Any]:
    receipts = list(job_root.rglob("pilot_receipt.json"))
    gates = list(job_root.rglob("recurrent_causal_gate.json"))
    validations = list(job_root.rglob("validation_metrics.json"))
    gradients = list(job_root.rglob("gradient_audit.json"))
    if not all(len(paths) == 1 for paths in (receipts, gates, validations, gradients)):
        raise ValueError(f"Incomplete pilot artifact set under {job_root}")
    receipt = read(receipts[0])
    gate = read(gates[0])
    validation = read(validations[0])
    gradient = read(gradients[0])
    summary = gate["summary"]
    clean = summary["clean"]
    by_task = validation.get("by_task") or {}
    recurrent_gap = float(validation.get("accuracy_gap_recurrent_minus_direct", -1.0))
    t3 = by_task.get("diagnose_misconception") or {}
    t3_gap = float(t3.get("accuracy_gap_recurrent_minus_direct", -1.0))
    remote_run_dir = str(receipt.get("remote_run_dir") or "")
    expected_run_dir = (
        "/mnt/amulet/studentmodeling/checkpoints/layerwise_recurrent_pilot_"
        f"{receipt.get('branch', '')}_seed42_n200_20260731"
    )
    expected_branch = EXPECTED_BRANCHES.get(str(receipt.get("branch")))
    checks = {
        "training_complete": receipt.get("status") == "pilot_training_complete",
        "exposure_complete": int(receipt.get("micro_steps", -1)) == 200
        and int(receipt.get("causal_pairs_seen", -1)) == 200,
        "registered_branch": expected_branch is not None
        and float(receipt.get("learning_rate", -1))
        == float(expected_branch["learning_rate"])
        and float(receipt.get("causal_weight", -1))
        == float(expected_branch["causal_weight"])
        and str(receipt.get("causal_objective_mode"))
        == str(expected_branch["causal_objective_mode"]),
        "layerwise": receipt.get("state_architecture") == "layerwise",
        "closed_decoder": receipt.get("raw_history_visible_to_decoder") is False
        and receipt.get("persistent_history_kv_cache") is False,
        "remote_pilot_artifacts": remote_run_dir == expected_run_dir,
        "gradient_reaches_lora": float((gradient.get("lora") or {}).get("gradient_abs_sum", 0)) > 0,
        "candidate_support": int(summary.get("clean_n", 0)) >= 50,
        "specificity": float(summary.get("D_specific", -1)) >= 0.15,
        "noop": float(clean["noop"]["flip_to_A_rate"]) <= 0.05,
        "same_rule": float(clean["samemal"]["flip_to_A_rate"]) <= 0.10,
        "validation_finite": all(
            value == value and abs(float(value)) < float("inf")
            for value in validation.values()
            if isinstance(value, (int, float))
        ),
        "task_gap": recurrent_gap >= -0.05,
        "T3_gap": t3_gap >= -0.02,
    }
    return {
        "branch": receipt["branch"],
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
        },
        "validation": validation,
        "checks": checks,
        "pass": all(checks.values()),
    }


def select(root: Path) -> dict[str, Any]:
    job_roots = sorted({path.parents[1] for path in root.rglob("metrics/pilot_receipt.json")})
    if len(job_roots) != 3:
        raise ValueError(f"Expected three pilot jobs, found {job_roots}")
    rows = [candidate(path) for path in job_roots]
    passing = [row for row in rows if row["pass"]]
    if not passing:
        return {
            "stage": "layerwise_recurrent_pilot_selection",
            "scientific_gate_pass": False,
            "selected_branch": None,
            "candidates": rows,
        }
    passing.sort(
        key=lambda row: (
            row["C4"]["D_specific"],
            row["C4"]["swap"],
            -row["C4"]["random"],
        ),
        reverse=True,
    )
    selected = passing[0]
    return {
        "stage": "layerwise_recurrent_pilot_selection",
        "scientific_gate_pass": True,
        "selected_branch": selected["branch"],
        "selected_artifact_root": selected["artifact_root"],
        "selected_C4": selected["C4"],
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
