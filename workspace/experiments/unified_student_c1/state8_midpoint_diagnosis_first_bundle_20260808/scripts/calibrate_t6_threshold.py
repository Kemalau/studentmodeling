#!/usr/bin/env python3
"""Fit a T6 threshold on calibration predictions, then apply it unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STAGE = "layerwise_t6_threshold_calibration_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any(row.get("task") != "predict_error_persistence" for row in rows):
        raise ValueError("Expected non-empty T6-only predictions")
    return rows


def metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    gold = [row["gold"] == "persistent" for row in rows]
    pred = [float(row["probabilities"]["persistent"]) >= threshold for row in rows]
    tp = sum(a and b for a, b in zip(gold, pred))
    fp = sum((not a) and b for a, b in zip(gold, pred))
    fn = sum(a and not b for a, b in zip(gold, pred))
    tn = sum((not a) and not b for a, b in zip(gold, pred))

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    return {
        "n": len(rows),
        "threshold": threshold,
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
        "positive_recall": tp / max(1, tp + fn),
        "positive_precision": tp / max(1, tp + fp),
        "false_positive_rate": fp / max(1, fp + tn),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def fit(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    probabilities = sorted({float(row["probabilities"]["persistent"]) for row in rows})
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(probabilities, probabilities[1:]))
    candidates.extend(probabilities)
    scored = [metrics(rows, threshold) for threshold in sorted(set(candidates))]
    # Macro-F1 is the frozen task gate.  Deterministic ties prefer the threshold
    # closest to 0.5, then the larger threshold (fewer false positives).
    best = max(scored, key=lambda row: (row["macro_f1"], -abs(row["threshold"] - 0.5), row["threshold"]))
    return float(best["threshold"]), best


def grouped(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    output = {"all": metrics(rows, threshold)}
    for source in sorted({str(row["source"]) for row in rows}):
        output[source] = metrics([row for row in rows if str(row["source"]) == source], threshold)
    return output


def fit_command(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.predictions)
    audit = json.loads(args.manifest_audit.read_text(encoding="utf-8"))
    ids = [str(row["example_id"]) for row in rows]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
    if audit.get("stage") != "layerwise_t6_calibration_manifest_v1":
        raise ValueError("Calibration manifest stage drift")
    if len(ids) != len(set(ids)):
        raise ValueError("Calibration predictions contain duplicate example IDs")
    if len(rows) != int(audit.get("selected_rows", -1)) or digest != audit.get("selected_example_id_set_sha256"):
        raise ValueError("Calibration predictions do not match the frozen manifest")
    if audit.get("candidate_training_overlap") != 0 or audit.get("formal_example_id_overlap") != 0:
        raise ValueError("Calibration manifest is not isolated")
    threshold, selected = fit(rows)
    result = {
        "stage": STAGE,
        "status": "frozen",
        "threshold_scope": "global_predict_error_persistence",
        "selection_metric": "macro_f1",
        "threshold": threshold,
        "calibration_predictions": str(args.predictions),
        "calibration_predictions_sha256": sha256(args.predictions),
        "calibration_manifest_audit": str(args.manifest_audit),
        "calibration_manifest_audit_sha256": sha256(args.manifest_audit),
        "checkpoint_hashes": json.loads(args.checkpoint_hashes.read_text(encoding="utf-8")),
        "default_metrics": grouped(rows, 0.5),
        "selected_metrics": grouped(rows, threshold),
        "selection": selected,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def apply_command(args: argparse.Namespace) -> None:
    frozen = json.loads(args.frozen_threshold.read_text(encoding="utf-8"))
    if frozen.get("stage") != STAGE or frozen.get("status") != "frozen":
        raise ValueError("Threshold artifact is not frozen")
    rows = read_jsonl(args.predictions)
    threshold = float(frozen["threshold"])
    result = {
        "stage": "layerwise_t6_calibrated_evaluation_v1",
        "threshold": threshold,
        "threshold_artifact": str(args.frozen_threshold),
        "threshold_artifact_sha256": sha256(args.frozen_threshold),
        "evaluation_predictions": str(args.predictions),
        "evaluation_predictions_sha256": sha256(args.predictions),
        "default_metrics": grouped(rows, 0.5),
        "calibrated_metrics": grouped(rows, threshold),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--predictions", type=Path, required=True)
    fit_parser.add_argument("--manifest-audit", type=Path, required=True)
    fit_parser.add_argument("--checkpoint-hashes", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.set_defaults(func=fit_command)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--frozen-threshold", type=Path, required=True)
    apply_parser.add_argument("--predictions", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.set_defaults(func=apply_command)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
