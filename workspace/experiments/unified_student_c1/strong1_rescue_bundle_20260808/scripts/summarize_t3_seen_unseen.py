#!/usr/bin/env python3
"""Split Eedi T3 generation quality by train-seen versus unseen taxonomy labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TAG = re.compile(r"^<misconception>.+</misconception>$", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object in {path}")
                rows.append(value)
    return rows


def unique_by_id(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id in result:
            raise ValueError(f"Missing or duplicate example_id in {path}: {example_id!r}")
        result[example_id] = row
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = [str(row["raw_output"]).strip() for row in rows]
    counts = Counter(outputs)
    judged = [row for row in rows if "llm_judge_correct" in row]
    return {
        "support": len(rows),
        "judged_support": len(judged),
        "semantic_accuracy": (
            sum(bool(row["llm_judge_correct"]) for row in judged) / len(judged)
            if judged
            else None
        ),
        "format_valid_rate": (
            sum(bool(TAG.fullmatch(output)) for output in outputs) / len(outputs)
            if outputs
            else None
        ),
        "unique_outputs": len(counts),
        "top_output_share": (
            counts.most_common(1)[0][1] / len(outputs) if outputs else None
        ),
    }


def build(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    train_labels = {
        str(row["target_label"])
        for row in train_rows
        if row.get("task") == "diagnose_misconception"
        and row.get("source") == "eedi_mining"
        and row.get("target_label") is not None
    }
    if not train_labels:
        raise ValueError("Training corpus has no labeled Eedi T3 rows")
    eval_by_id = {
        key: row
        for key, row in unique_by_id(eval_rows, Path("eval_jsonl")).items()
        if row.get("task") == "diagnose_misconception"
        and row.get("source") == "eedi_mining"
    }
    prediction_by_id = unique_by_id(predictions, Path("predictions_jsonl"))
    judgment_by_id = unique_by_id(judgments, Path("judgments_jsonl"))

    joined: list[dict[str, Any]] = []
    for example_id in sorted(set(prediction_by_id) & set(eval_by_id)):
        if example_id not in judgment_by_id:
            raise ValueError(f"Eedi prediction lacks semantic judgment: {example_id}")
        eval_row = eval_by_id[example_id]
        judgment = judgment_by_id[example_id]
        if "llm_judge_correct" not in judgment:
            raise ValueError(f"Unresolved semantic judgment: {example_id}")
        joined.append(
            {
                "example_id": example_id,
                "target_label": str(eval_row.get("target_label")),
                "raw_output": str(prediction_by_id[example_id].get("raw_output", "")),
                "llm_judge_correct": bool(judgment["llm_judge_correct"]),
            }
        )
    if not joined:
        raise ValueError("No Eedi T3 predictions joined to evaluation rows")

    seen = [row for row in joined if row["target_label"] in train_labels]
    unseen = [row for row in joined if row["target_label"] not in train_labels]
    return {
        "stage": "eedi_t3_seen_unseen_taxonomy_diagnostic_v1",
        "diagnostic_only": True,
        "selection_role": "diagnose_open_taxonomy_generalization",
        "train_eedi_labels": len(train_labels),
        "evaluation_eedi_rows": len(joined),
        "evaluation_seen_label_rows": len(seen),
        "evaluation_unseen_label_rows": len(unseen),
        "seen_label_fraction": len(seen) / len(joined),
        "all": summarize(joined),
        "seen_label": summarize(seen),
        "unseen_label": summarize(unseen),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--judgments-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        read_jsonl(args.train_jsonl),
        read_jsonl(args.eval_jsonl),
        read_jsonl(args.predictions_jsonl),
        read_jsonl(args.judgments_jsonl),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
