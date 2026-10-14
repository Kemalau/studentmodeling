#!/usr/bin/env python3
"""Audit whether a future-profile task is identifiable from its visible history.

This is a data-only audit.  It never loads a model and never fits on test
labels.  In addition to coverage, it reports a deliberately simple sanity
control: Beta(1,1) KC rates computed from the exact updater-visible history are
compared with future rates, then student ownership is permuted.  A strong
seen-KC ownership control alongside weak all-KC coverage distinguishes an
information-boundary problem from absence of any student-specific signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_matrices(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vocabulary = [str(value) for value in rows[0]["kc_vocabulary"]]
    index = {name: position for position, name in enumerate(vocabulary)}
    rates = np.empty((len(rows), len(vocabulary)), dtype=np.float64)
    support = np.zeros_like(rates, dtype=bool)
    for row_index, row in enumerate(rows):
        counts = np.zeros(len(vocabulary), dtype=np.float64)
        correct = np.zeros(len(vocabulary), dtype=np.float64)
        for event in row["state_history"]:
            for name in set(str(value) for value in event["kc_names"]):
                if name not in index:
                    continue
                position = index[name]
                counts[position] += 1
                correct[position] += float(event["correctness"])
        rates[row_index] = (correct + 1.0) / (counts + 2.0)
        support[row_index] = counts > 0
    return rates, support, np.asarray(vocabulary, dtype=str)


def future_matrices(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    rates = np.asarray(
        [
            [0.5 if value is None else float(value) for value in row["future_smoothed_rates"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    mask = np.asarray([row["future_eligible_mask"] for row in rows], dtype=bool)
    return rates, mask


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        raise ValueError("RMSE mask is empty")
    return float(np.sqrt(np.square(prediction[mask] - target[mask]).mean()))


def ownership_null(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0 or len(prediction) < 2:
        raise ValueError("Ownership null needs rows and positive replicates")
    native = masked_rmse(prediction, target, mask)
    rng = np.random.default_rng(seed)
    identity = np.arange(len(prediction))
    null = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        permutation = rng.permutation(len(prediction))
        if np.array_equal(permutation, identity):
            permutation = np.roll(permutation, 1)
        null[replicate] = masked_rmse(prediction[permutation], target, mask)
    return {
        "native_rmse": native,
        "permuted_rmse_mean": float(null.mean()),
        "permuted_rmse_ci95": [
            float(value) for value in np.quantile(null, [0.025, 0.975])
        ],
        "delta_permuted_mean_minus_native": float(null.mean() - native),
        "p_one_sided": float((1 + np.sum(null <= native)) / (replicates + 1)),
        "replicates": replicates,
    }


def audit_split(path: Path, *, replicates: int, seed: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty manifest split: {path}")
    visible_rate, visible_support, vocabulary = visible_matrices(rows)
    future_rate, future_support = future_matrices(rows)
    if visible_rate.shape != future_rate.shape:
        raise ValueError("Visible and future KC widths differ")
    seen = future_support & visible_support
    unseen = future_support & ~visible_support
    weakness_coverage = []
    no_weakness_seen = []
    for row, support in zip(rows, visible_support, strict=True):
        weak = {str(value) for value in row["weak_kcs"]}
        visible = set(vocabulary[support])
        overlap = len(weak & visible)
        weakness_coverage.append(overlap / len(weak))
        no_weakness_seen.append(overlap == 0)
    cutoff = np.asarray([int(row["cutoff_position"]) for row in rows])
    history = np.asarray([int(row["state_history_length"]) for row in rows])
    result = {
        "rows": len(rows),
        "sha256": sha256(path),
        "cutoff_position": {
            "minimum": int(cutoff.min()),
            "median": float(np.median(cutoff)),
            "maximum": int(cutoff.max()),
        },
        "visible_history_length": {
            "minimum": int(history.min()),
            "median": float(np.median(history)),
            "maximum": int(history.max()),
            "mean_fraction_of_prefix": float(np.mean(history / cutoff)),
        },
        "future_supported_cells": int(future_support.sum()),
        "seen_future_supported_cells": int(seen.sum()),
        "unseen_future_supported_cells": int(unseen.sum()),
        "seen_fraction_of_future_supported_cells": float(
            seen.sum() / future_support.sum()
        ),
        "future_weakness_visible_history_coverage": float(np.mean(weakness_coverage)),
        "students_with_no_future_weakness_seen": int(sum(no_weakness_seen)),
        "students_with_no_future_weakness_seen_rate": float(np.mean(no_weakness_seen)),
        "visible_history_rate_ownership_sanity": ownership_null(
            visible_rate,
            future_rate,
            seen,
            replicates=replicates,
            seed=seed,
        ),
        "ownership_sanity_scope": (
            "future-supported KCs also observed in the exact updater-visible history"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutation-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    manifest_path = args.manifest_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {
        split: audit_split(
            args.manifest_dir / f"{split}.jsonl",
            replicates=args.permutation_replicates,
            seed=args.seed + index,
        )
        for index, split in enumerate(("train", "valid", "test"))
    }
    result = {
        "stage": "state8_profile_information_boundary_audit_v1",
        "status": "complete",
        "dataset_source": manifest.get("source"),
        "manifest_dir": str(args.manifest_dir),
        "manifest_sha256": sha256(manifest_path),
        "model_loaded": False,
        "test_labels_used_for_fitting": False,
        "splits": splits,
        "interpretation_contract": {
            "coverage": "identifiability diagnostic, not a model score",
            "visible_history_rate_ownership_sanity": (
                "data-only positive-control baseline, not State8 evidence"
            ),
            "unseen_kcs": (
                "future transfer beyond directly observed KC evidence; keep separate "
                "from observed-support ownership"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
