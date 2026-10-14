#!/usr/bin/env python3
"""Attach a frozen validation-selected T6 threshold to one exact candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE = "layerwise_t6_calibrated_candidate_v1"
REFERENCE_MACRO_F1 = 0.5959866676584252
MIN_RATIO = 0.85


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    summary: dict[str, Any],
    frozen: dict[str, Any],
    formal: dict[str, Any],
    completion: dict[str, Any],
    *,
    summary_sha256: str,
    frozen_sha256: str,
) -> dict[str, Any]:
    candidate_id = "eedi_full_open_t3__state_path__lr2e-06"
    if (
        summary.get("candidate_id") != candidate_id
        or summary.get("branch") != "eedi_full_open_t3"
        or summary.get("loss_profile") != "state_path"
        or float(summary.get("learning_rate", -1)) != 2e-6
    ):
        raise ValueError("Calibrated candidate identity drift")
    if summary.get("checks", {}).get("C4") is not True or summary.get("checks", {}).get("T3_semantic") is not True:
        raise ValueError("Calibration cannot rescue a noncausal or T3-failing candidate")
    if frozen.get("stage") != "layerwise_t6_threshold_calibration_v1" or frozen.get("status") != "frozen":
        raise ValueError("T6 threshold is not frozen")
    if frozen.get("checkpoint_hashes", {}).get("candidate_id") != candidate_id:
        raise ValueError("Threshold candidate drift")
    hashes = completion.get("sha256") or {}
    if (
        frozen.get("checkpoint_hashes", {}).get("adapter_model_sha256") != hashes.get("adapter_model")
        or frozen.get("checkpoint_hashes", {}).get("recurrent_state_sha256") != hashes.get("recurrent_state")
    ):
        raise ValueError("Threshold checkpoint hash drift")
    if formal.get("stage") != "layerwise_t6_calibrated_evaluation_v1":
        raise ValueError("Calibrated formal evidence stage drift")
    if float(formal.get("threshold", -1)) != float(frozen.get("threshold", -2)):
        raise ValueError("Formal evaluation did not use the frozen threshold")
    if formal.get("threshold_artifact_sha256") != frozen_sha256:
        raise ValueError("Formal threshold artifact hash drift")
    calibrated = formal.get("calibrated_metrics", {}).get("all") or {}
    support = int(calibrated.get("n", -1))
    macro_f1 = float(calibrated.get("macro_f1", -1))
    recall = float(calibrated.get("positive_recall", -1))
    retention = macro_f1 / REFERENCE_MACRO_F1
    if support != int(summary.get("T6", {}).get("support", -2)) or support != 976:
        raise ValueError("Calibrated T6 support drift")
    if retention < MIN_RATIO or recall <= 0:
        raise ValueError("Frozen calibration does not pass the T6 gate")

    checks = dict(summary["checks"])
    checks["T6_retention"] = True
    calibrated_summary = dict(summary)
    calibrated_summary.update(
        stage=STAGE,
        checks=checks,
        all_gates_pass=all(checks.values()),
        source_candidate_summary_sha256=summary_sha256,
        T6={
            **summary["T6"],
            "decision_threshold": float(frozen["threshold"]),
            "macro_f1": macro_f1,
            "retention_ratio": retention,
            "positive_recall": recall,
            "calibrated": True,
            "default_threshold_metrics": formal.get("default_metrics", {}).get("all"),
            "calibrated_metrics": calibrated,
        },
        t6_calibration={
            "stage": frozen["stage"],
            "artifact_sha256": frozen_sha256,
            "decision_threshold": float(frozen["threshold"]),
            "selection_split": "unused_original_validation",
            "selection_rows": int(frozen["selection"]["n"]),
            "candidate_training_overlap": 0,
            "formal_example_id_overlap": 0,
            "formal_evidence_sha256": formal.get("evaluation_predictions_sha256"),
        },
    )
    if not calibrated_summary["all_gates_pass"]:
        raise ValueError("Calibrated candidate still fails a frozen gate")
    return calibrated_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--frozen-threshold", type=Path, required=True)
    parser.add_argument("--formal-calibrated", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(
        read(args.candidate_summary),
        read(args.frozen_threshold),
        read(args.formal_calibrated),
        read(args.completion),
        summary_sha256=sha256(args.candidate_summary),
        frozen_sha256=sha256(args.frozen_threshold),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
