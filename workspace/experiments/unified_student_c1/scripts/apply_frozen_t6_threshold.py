#!/usr/bin/env python3
"""Apply a preregistered T6 threshold to evaluator artifacts in place."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from evaluate_formal_protocols import redacted_records, summarize, write_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    threshold = float(receipt.get("t6_decision_threshold", -1))
    artifact_sha256 = str(receipt.get("t6_calibration_artifact_sha256", ""))
    if not 0 < threshold < 1 or len(artifact_sha256) != 64:
        raise ValueError("Final-fit receipt lacks a frozen T6 calibration")
    predictions = args.output_dir / "predictions.jsonl"
    redacted = args.output_dir / "predictions_redacted.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line]
    t6_rows = [row for row in rows if row.get("task") == "predict_error_persistence"]
    if not t6_rows:
        raise ValueError("No T6 rows found for frozen-threshold application")
    before_sha256 = sha256(predictions)
    for row in t6_rows:
        probability = float(row["probabilities"]["persistent"])
        row["prediction"] = "persistent" if probability >= threshold else "transient"
    default_metrics = args.output_dir / "metrics_default_threshold.json"
    default_predictions = args.output_dir / "predictions_default_threshold.jsonl"
    default_redacted = args.output_dir / "predictions_redacted_default_threshold.jsonl"
    shutil.copy2(metrics_path, default_metrics)
    shutil.copy2(predictions, default_predictions)
    shutil.copy2(redacted, default_redacted)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    updated_groups = summarize(rows)
    for name, value in updated_groups.items():
        if name.startswith("predict_error_persistence/"):
            metrics["metrics_by_group"][name] = value
    metrics["t6_decision_threshold"] = threshold
    metrics["t6_calibration_artifact_sha256"] = artifact_sha256
    metrics["t6_threshold_selection_data"] = "unused_original_validation"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write_jsonl(predictions, rows)
    write_jsonl(redacted, redacted_records(rows))
    audit = {
        "stage": "layerwise_t6_frozen_threshold_application_v1",
        "threshold": threshold,
        "calibration_artifact_sha256": artifact_sha256,
        "rows_updated": len(t6_rows),
        "predictions_before_sha256": before_sha256,
        "predictions_after_sha256": sha256(predictions),
        "default_metrics_sha256": sha256(default_metrics),
        "calibrated_metrics_sha256": sha256(metrics_path),
    }
    (args.output_dir / "t6_threshold_application.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
