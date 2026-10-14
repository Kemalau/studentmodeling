#!/usr/bin/env python3
"""Train pyBKT on temporal T1 sources using each prompt history as context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyBKT.models import Model
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


DEFAULT_SOURCES = {"dbe_kt22", "xes3g5m", "malrule"}


def read(path: Path, sources: set[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "")
            if row.get("task") == "predict_correctness" and source in sources:
                rows.append(row)
    return rows


def source(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("mix_source") or row.get("dataset") or "unknown")


def event_skill(row: dict[str, Any], event: dict[str, Any]) -> str:
    values = event.get("kc_ids") or []
    value = "|".join(str(item) for item in values) or str(event.get("skill_id") or event.get("family") or "unknown")
    return f"{source(row)}::{value}"


def event_question(row: dict[str, Any], event: dict[str, Any]) -> str:
    value = event.get("question_id") or event.get("surface_id") or "unknown"
    return f"{source(row)}::{value}"


def to_frame(rows: list[dict[str, Any]], split: str) -> pd.DataFrame:
    records = []
    for row_index, row in enumerate(rows):
        sequence_id = f"{split}:{row_index}"
        for order_id, event in enumerate(row.get("history") or []):
            records.append(
                {
                    "user_id": sequence_id,
                    "skill_name": event_skill(row, event),
                    "problem_id": event_question(row, event),
                    "correct": int(event.get("correctness", event.get("is_correct", 0))),
                    "order_id": order_id,
                    "target_marker": 0,
                    "row_id": row_index,
                    "source": source(row),
                }
            )
        query = row.get("query") or {}
        records.append(
            {
                "user_id": sequence_id,
                "skill_name": f"{source(row)}::{(row.get('metadata') or {}).get('skill_id', 'unknown')}",
                "problem_id": f"{source(row)}::{(row.get('metadata') or {}).get('question_id', 'unknown')}",
                "correct": int(row["target_label"] == "correct"),
                "order_id": len(row.get("history") or []),
                "target_marker": 1,
                "row_id": row_index,
                "source": source(row),
            }
        )
    return pd.DataFrame.from_records(records)


def score(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["correct"].to_numpy(dtype=int)
    probabilities = frame["correct_predictions"].to_numpy(dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    result["by_source"] = {
        source_name: score_without_groups(group)
        for source_name, group in frame.groupby("source", sort=True)
    }
    return result


def score_without_groups(frame: pd.DataFrame) -> dict[str, Any]:
    labels = frame["correct"].to_numpy(dtype=int)
    probabilities = frame["correct_predictions"].to_numpy(dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(set(labels.tolist())) == 2 else None,
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", default=sorted(DEFAULT_SOURCES))
    parser.add_argument("--num-fits", type=int, default=1)
    args = parser.parse_args()

    sources = set(args.sources)
    train_rows = read(args.train_jsonl, sources)
    test_rows = read(args.test_jsonl, sources)
    train_frame = to_frame(train_rows, "train")
    test_frame = to_frame(test_rows, "test")
    model = Model(seed=42, num_fits=args.num_fits, parallel=True)
    model.fit(data=train_frame)
    known_skills = set(model.fit_model)
    known_test = test_frame[test_frame["skill_name"].isin(known_skills)].copy()
    predicted = model.predict(data=known_test)
    known_targets = predicted[predicted["target_marker"] == 1][["row_id", "correct_predictions"]].copy()
    targets = test_frame[test_frame["target_marker"] == 1].copy()
    targets = targets.merge(known_targets, on="row_id", how="left")
    source_rates = train_frame[train_frame["target_marker"] == 1].groupby("source")["correct"].mean().to_dict()
    global_rate = float(train_frame[train_frame["target_marker"] == 1]["correct"].mean())
    fallback_mask = targets["correct_predictions"].isna()
    targets.loc[fallback_mask, "correct_predictions"] = targets.loc[fallback_mask, "source"].map(source_rates).fillna(global_rate)
    result = {
        "sources": sorted(sources),
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "train_interactions": len(train_frame),
        "test_interactions": len(test_frame),
        "known_skills": len(known_skills),
        "cold_skill_fallback_targets": int(fallback_mask.sum()),
        "metrics": score(targets),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
