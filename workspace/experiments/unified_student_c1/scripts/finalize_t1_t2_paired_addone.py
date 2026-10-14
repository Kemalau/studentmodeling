#!/usr/bin/env python3
"""Audit and summarize the formal T1 evaluation of the paired-T2 add-one run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from compare_taskonly_unified_paired import compare, read_jsonl
except ModuleNotFoundError:  # Imported as scripts.finalize_t1_t2_paired_addone in tests.
    from scripts.compare_taskonly_unified_paired import compare, read_jsonl


EXPECTED_ROWS = 10_913
EXPECTED_EVAL_SHA256 = "bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8"
T1_ONLY_ACCURACY = 0.691652157976725
UNIFIED_ACCURACY = 0.6875286355722533


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(paths: list[Path], label: str) -> Path:
    require(len(paths) == 1, f"expected one {label}, found {paths}")
    return paths[0]


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    require(trials > 0, "Wilson interval needs nonzero support")
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2)) / denominator
    return [centre - radius, centre + radius]


def binary_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute tie-aware binary AUROC from average ranks (positive label is one)."""
    require(len(labels) == len(scores) and labels, "AUROC inputs must be aligned and nonempty")
    positives = sum(labels)
    negatives = len(labels) - positives
    require(positives > 0 and negatives > 0, "AUROC requires both labels")
    ordered = sorted(zip(scores, labels), key=lambda value: value[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def challenger_view(raw: dict[str, Any], reference: str) -> dict[str, Any]:
    """Rename the task-only/unified helper's roles for an add-one challenger."""
    discordant = raw["discordant"]
    return {
        "challenger": "T1 + paired T2 add-one",
        "reference": reference,
        "n": raw["n"],
        "challenger_primary": raw["taskonly_primary"],
        "reference_primary": raw["unified_primary"],
        "challenger_minus_reference": raw["taskonly_minus_unified"],
        "paired_bootstrap_95ci": raw["paired_bootstrap_95ci"],
        "discordant": {
            "challenger_only_correct": discordant["taskonly_only_correct"],
            "reference_only_correct": discordant["unified_only_correct"],
            "both_correct": discordant["both_correct"],
            "both_wrong": discordant["both_wrong"],
        },
        "mcnemar_exact_two_sided_p": raw["mcnemar_exact_two_sided_p"],
        "example_id_set_sha256": raw["example_id_set_sha256"],
    }


def load_stage_receipt(download: Path, stage: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in download.rglob("completion.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("stage") == stage:
            matches.append((path, value))
    require(len(matches) == 1, f"expected one {stage} receipt, found {[path for path, _ in matches]}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    metrics_path = one(
        [path for path in args.download.rglob("metrics.json") if "eval_t1" in str(path)],
        "eval_t1 metrics.json",
    )
    predictions_path = metrics_path.parent / "predictions.jsonl"
    require(predictions_path.is_file(), f"missing {predictions_path}")
    train_receipt_path, train_receipt = load_stage_receipt(
        args.download, "t1_t2_paired_addone_train_v1"
    )
    eval_receipt_path, eval_receipt = load_stage_receipt(
        args.download, "t1_t2_paired_addone_t1_eval_v1"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = read_jsonl(predictions_path)

    require(train_receipt.get("status") == "complete", "training receipt is not complete")
    require(eval_receipt.get("status") == "complete", "evaluation receipt is not complete")
    require(train_receipt.get("seed") == 42, "training seed drift")
    require(train_receipt.get("train_rows") == 162_765, "training support drift")
    require(train_receipt.get("t1_rows") == 110_525, "T1 exposure drift")
    require(train_receipt.get("paired_t2_rows") == 52_240, "paired-T2 support drift")
    require(train_receipt.get("optimizer_steps") == 40_692, "optimizer horizon drift")
    require(eval_receipt.get("evaluation_rows") == EXPECTED_ROWS, "evaluation receipt support drift")
    require(metrics.get("artifact_sha256", {}).get("evaluation_input") == EXPECTED_EVAL_SHA256,
            "formal evaluation input hash drift")
    adapter_sha256 = metrics.get("artifact_sha256", {}).get("adapter_model")
    require(adapter_sha256 == train_receipt.get("adapter_sha256"), "adapter lineage hash mismatch")
    require(adapter_sha256 == eval_receipt.get("adapter_sha256"), "evaluation adapter hash mismatch")
    require(sha256(predictions_path) == eval_receipt.get("predictions_sha256"),
            "prediction artifact hash mismatch")
    require(len(rows) == EXPECTED_ROWS, f"prediction support drift: {len(rows)}")
    ids = [str(row["example_id"]) for row in rows]
    require(len(set(ids)) == EXPECTED_ROWS, "duplicate T1 example IDs")
    require({row.get("task") for row in rows} == {"predict_correctness"}, "task drift")

    correct = [str(row.get("prediction")) == str(row.get("gold")) for row in rows]
    accuracy = sum(correct) / EXPECTED_ROWS
    weighted_accuracy = sum(
        int(group["n"]) * float(group["accuracy"])
        for group in metrics["metrics_by_group"].values()
    ) / EXPECTED_ROWS
    require(abs(accuracy - weighted_accuracy) < 1e-12, "prediction/metric accuracy mismatch")
    labels = [int(str(row.get("gold")) == "correct") for row in rows]
    scores = [float(row["probabilities"]["correct"]) for row in rows]
    auroc = binary_auroc(labels, scores)

    project = Path(__file__).resolve().parents[1]
    formal = project / "results" / "taskonly_qwen_formal_eval_20260813"
    taskonly_path = one(list((formal / "T1").rglob("eval/T1/predictions.jsonl")), "T1-only predictions")
    unified_path = one(
        list((project / "results" / "formal_specialist_v1" / "qwen_amlt").rglob(
            "adapter_classification/predictions_redacted.jsonl"
        )),
        "Unified classification predictions",
    )
    taskonly_rows = read_jsonl(taskonly_path)
    unified_rows = read_jsonl(unified_path)
    versus_taskonly_raw = compare(
        taskonly_rows, rows, "T1", bootstrap_samples=args.bootstrap_samples, seed=42
    )
    versus_unified_raw = compare(
        unified_rows, rows, "T1", bootstrap_samples=args.bootstrap_samples, seed=43
    )
    require(abs(versus_taskonly_raw["unified_primary"] - T1_ONLY_ACCURACY) < 1e-12,
            "T1-only baseline drift")
    require(abs(versus_unified_raw["unified_primary"] - UNIFIED_ACCURACY) < 1e-12,
            "Unified baseline drift")
    versus_taskonly = challenger_view(versus_taskonly_raw, "T1-only")
    versus_unified = challenger_view(versus_unified_raw, "Unified")

    output = {
        "status": "complete",
        "protocol": "T1 + interaction-matched T2 add-one; T1 exposure matched, total updates not matched",
        "interpretation_scope": "diagnostic add-one transfer; not a total-update-matched causal comparison",
        "n": EXPECTED_ROWS,
        "t1_t2_accuracy": accuracy,
        "t1_t2_accuracy_wilson_95ci": wilson_interval(sum(correct), EXPECTED_ROWS),
        "t1_t2_auroc": auroc,
        "t1_only_accuracy": T1_ONLY_ACCURACY,
        "unified_accuracy": UNIFIED_ACCURACY,
        "t1_t2_minus_t1_only": accuracy - T1_ONLY_ACCURACY,
        "t1_t2_minus_unified": accuracy - UNIFIED_ACCURACY,
        "versus_t1_only_paired": versus_taskonly,
        "versus_unified_paired": versus_unified,
        "lineage": {
            "adapter_sha256": adapter_sha256,
            "evaluation_input_sha256": EXPECTED_EVAL_SHA256,
            "predictions_sha256": sha256(predictions_path),
            "example_id_set_sha256": versus_taskonly["example_id_set_sha256"],
            "metrics_path": str(metrics_path),
            "predictions_path": str(predictions_path),
            "training_receipt_path": str(train_receipt_path),
            "evaluation_receipt_path": str(eval_receipt_path),
        },
    }
    args.state.mkdir(parents=True, exist_ok=True)
    (args.state / "comparison.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    taskonly_ci = versus_taskonly["paired_bootstrap_95ci"]
    unified_ci = versus_unified["paired_bootstrap_95ci"]
    (args.state / "comparison.md").write_text(
        "# T1 + paired T2 add-one result\n\n"
        f"- T1+T2 accuracy: {accuracy:.6f} "
        f"(Wilson 95% CI [{output['t1_t2_accuracy_wilson_95ci'][0]:.6f}, "
        f"{output['t1_t2_accuracy_wilson_95ci'][1]:.6f}]); AUROC: {auroc:.6f}\n"
        f"- T1-only accuracy: {T1_ONLY_ACCURACY:.6f}\n"
        f"- Unified accuracy: {UNIFIED_ACCURACY:.6f}\n"
        f"- T1+T2 minus T1-only: {output['t1_t2_minus_t1_only']:+.6f}; "
        f"paired 95% CI [{taskonly_ci[0]:+.6f}, {taskonly_ci[1]:+.6f}]; "
        f"McNemar p={versus_taskonly['mcnemar_exact_two_sided_p']:.6g}\n"
        f"- T1+T2 minus Unified: {output['t1_t2_minus_unified']:+.6f}; "
        f"paired 95% CI [{unified_ci[0]:+.6f}, {unified_ci[1]:+.6f}]; "
        f"McNemar p={versus_unified['mcnemar_exact_two_sided_p']:.6g}\n\n"
        "Protocol note: T1 exposure is matched to T1-only, but the added paired T2 rows increase "
        "the total number of updates. This is an add-one transfer diagnostic, not a total-update-matched "
        "causal comparison.\n",
        encoding="utf-8",
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
