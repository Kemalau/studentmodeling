#!/usr/bin/env python3
"""Build diagnosis-only pairwise semantic controls from a C4 generation panel.

Gold-reference judgments answer whether a generated diagnosis is correct.  They
cannot, by themselves, answer whether a no-op or same-rule intervention kept
the *model's own* diagnosis unchanged.  This module creates a second, strictly
diagnosis-only panel whose reference is the native model output for the same
pair.  Schema quality remains a separate task metric and therefore does not
confound the intervention-invariance measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from t3_judge_utils import extract_diagnosis, write_jsonl
except ImportError:  # package-style tests
    from scripts.t3_judge_utils import extract_diagnosis, write_jsonl


CONDITIONS = ("noop", "samemal", "random", "swap")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_panel(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_idx = int(row["pair_idx"])
        condition = str(row["condition"])
        if condition in grouped[pair_idx]:
            raise ValueError(f"Duplicate C4 generation pair={pair_idx} condition={condition}")
        grouped[pair_idx][condition] = row
    if not grouped:
        raise ValueError("Pairwise C4 panel is empty")

    predictions: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    excluded: list[int] = []
    for pair_idx, conditions in sorted(grouped.items()):
        expected = {"native", *CONDITIONS}
        if set(conditions) != expected:
            raise ValueError(
                f"C4 pair {pair_idx} condition drift: {sorted(conditions)}"
            )
        native = conditions["native"]
        native_diagnosis = extract_diagnosis(str(native.get("raw_output") or ""))
        if not native_diagnosis:
            excluded.append(pair_idx)
            continue
        # A legacy <misconception> target intentionally disables the structured
        # schema gate in the generic semantic judge.  This endpoint measures
        # diagnosis invariance only; schema validity is reported by T3/C4 gold
        # judging and must not be counted twice here.
        reference = f"<misconception>{native_diagnosis}</misconception>"
        for condition in CONDITIONS:
            row = conditions[condition]
            example_id = f"pairwise-diagnosis-c4:{pair_idx:04d}:{condition}:native-output"
            predictions.append(
                {
                    "example_id": example_id,
                    "task": "diagnose_misconception",
                    "source": "c4_pairwise_control",
                    "protocol": "closed_state_pairwise_output_invariance_v1",
                    "pair_idx": pair_idx,
                    "condition": condition,
                    "reference_role": "native_output",
                    "target": reference,
                    "raw_output": str(row.get("raw_output") or ""),
                }
            )
            evaluation.append(
                {
                    "example_id": example_id,
                    "task": "diagnose_misconception",
                    "source": "c4_pairwise_control",
                    "direct_text": str(
                        row.get("decode_prompt") or native.get("decode_prompt") or ""
                    ),
                }
            )
    eligible = len(grouped) - len(excluded)
    if eligible <= 0:
        raise ValueError("No C4 pair has an interpretable native diagnosis")
    receipt = {
        "stage": "state8_diagnosis_c4_pairwise_panel_v1",
        "status": "complete",
        "conditions": list(CONDITIONS),
        "input_pairs": len(grouped),
        "eligible_native_pairs": eligible,
        "excluded_native_pairs": excluded,
        "prediction_rows": len(predictions),
        "schema_authoritative": False,
        "semantic_endpoint": "diagnosis_invariance_to_native_model_output",
        "gold_task_correctness_measured_separately": True,
    }
    return predictions, evaluation, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    predictions, evaluation, receipt = build_panel(
        read_jsonl(args.generations_jsonl)
    )
    write_jsonl(args.predictions_jsonl, predictions)
    write_jsonl(args.eval_jsonl, evaluation)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
