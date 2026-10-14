#!/usr/bin/env python3
"""Summarize the confirmatory ReKT comparison on the frozen five-source T1 test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


GROUPS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "overall": lambda row: True,
    "real_all": lambda row: row["source"] != "malrule",
    "real_temporal": lambda row: row["source"] in {"dbe_kt22", "foundationalassist", "xes3g5m"},
}


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return {row["example_id"]: row for row in (json.loads(line) for line in handle if line.strip())}


def digest_ids(ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def probability(row: dict[str, Any]) -> float:
    if "p_correct" in row:
        return float(row["p_correct"])
    return float(row["probabilities"]["correct"])


def bootstrap_accuracy_delta(delta: np.ndarray, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(delta)
    values = np.empty(replicates, dtype=float)
    for index in range(replicates):
        values[index] = delta[rng.integers(0, n, n)].mean()
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def metrics(gold: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(gold, probabilities >= 0.5)),
        "auroc": float(roc_auc_score(gold, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-predictions", type=Path, required=True)
    parser.add_argument("--rekt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_824)
    args = parser.parse_args()

    unified = {
        key: row
        for key, row in read_jsonl(args.unified_predictions).items()
        if row.get("task") == "predict_correctness"
    }
    paths = sorted(args.rekt_root.rglob("predictions.jsonl"))
    if len(paths) != 3:
        raise ValueError(f"Expected three ReKT prediction files, found {paths}")
    rekt = [read_jsonl(path) for path in paths]
    ids = sorted(unified)
    if len(ids) != 10_913:
        raise ValueError(f"Unified support drift: {len(ids)}")
    for rows in rekt:
        if set(rows) != set(ids):
            raise ValueError("ReKT/Unified example-ID mismatch")
        if any((rows[key]["gold"] == "correct") != (unified[key]["gold"] == "correct") for key in ids):
            raise ValueError("ReKT/Unified gold-label mismatch")

    payload: dict[str, Any] = {
        "stage": "rekt_t1_confirmatory_v1",
        "status": "complete",
        "support": len(ids),
        "example_id_set_sha256": digest_ids(ids),
        "seeds": [],
        "aggregate": {},
        "ensemble": {},
        "protocol": {
            "split": "frozen five-source T1",
            "target_response_exposed": False,
            "groups": list(GROUPS),
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }
    rekt_probabilities = np.asarray([[probability(rows[key]) for key in ids] for rows in rekt])
    ensemble_probabilities = rekt_probabilities.mean(axis=0)
    for seed_index, (path, rows) in enumerate(zip(paths, rekt)):
        record: dict[str, Any] = {"path": str(path), "groups": {}}
        metrics_path = next(path.parent.glob("metrics.json"))
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        record["seed"] = int(metadata["seed"])
        for group_name, keep in GROUPS.items():
            group_ids = [key for key in ids if keep(unified[key])]
            gold = np.asarray([unified[key]["gold"] == "correct" for key in group_ids], dtype=np.int64)
            unified_probability = np.asarray([probability(unified[key]) for key in group_ids])
            rekt_probability = np.asarray([probability(rows[key]) for key in group_ids])
            unified_correct = (unified_probability >= 0.5) == gold
            rekt_correct = (rekt_probability >= 0.5) == gold
            delta = unified_correct.astype(np.int8) - rekt_correct.astype(np.int8)
            record["groups"][group_name] = {
                "n": len(group_ids),
                "unified": metrics(gold, unified_probability),
                "rekt": metrics(gold, rekt_probability),
                "unified_minus_rekt_accuracy": float(delta.mean()),
                "paired_bootstrap_accuracy_95ci": bootstrap_accuracy_delta(
                    delta, args.bootstrap_seed + seed_index, args.bootstrap_replicates
                ),
                "unified_only_correct": int(np.sum(unified_correct & ~rekt_correct)),
                "rekt_only_correct": int(np.sum(~unified_correct & rekt_correct)),
            }
        payload["seeds"].append(record)

    payload["seeds"].sort(key=lambda record: record["seed"])
    for group_name in GROUPS:
        payload["aggregate"][group_name] = {}
        for metric_name in ("accuracy", "auroc"):
            values = np.asarray(
                [record["groups"][group_name]["rekt"][metric_name] for record in payload["seeds"]]
            )
            payload["aggregate"][group_name][metric_name] = {
                "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)),
            }

    for group_name, keep in GROUPS.items():
        indices = np.asarray([keep(unified[key]) for key in ids])
        gold = np.asarray([unified[key]["gold"] == "correct" for key in ids], dtype=np.int64)[indices]
        unified_probability = np.asarray([probability(unified[key]) for key in ids])[indices]
        rekt_probability = ensemble_probabilities[indices]
        unified_correct = (unified_probability >= 0.5) == gold
        rekt_correct = (rekt_probability >= 0.5) == gold
        delta = unified_correct.astype(np.int8) - rekt_correct.astype(np.int8)
        payload["ensemble"][group_name] = {
            "n": int(indices.sum()),
            "unified": metrics(gold, unified_probability),
            "rekt": metrics(gold, rekt_probability),
            "unified_minus_rekt_accuracy": float(delta.mean()),
            "paired_bootstrap_accuracy_95ci": bootstrap_accuracy_delta(
                delta, args.bootstrap_seed + 100, args.bootstrap_replicates
            ),
            "unified_only_correct": int(np.sum(unified_correct & ~rekt_correct)),
            "rekt_only_correct": int(np.sum(~unified_correct & rekt_correct)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
