#!/usr/bin/env python3
"""Independently audit State8 future-profile utility by observed KC support.

The frozen State8 readout is fit on train students, ridge alpha is selected on
validation students, and the selected readout is refit on train+validation.
Test labels are used only for scoring.  The primary output separates future-KC
cells whose KC was named in the exact updater-visible history from cells whose
KC was absent from that history.  It also evaluates the same-window KC summary
and train/validation population controls under exactly the same masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "valid", "test")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (x - mean) / scale, mean, scale


def normalize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) - mean) / scale


def fit_weighted_ridge(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Match the preregistered one-head-per-KC masked ridge protocol."""
    weights: list[np.ndarray] = []
    biases: list[float] = []
    supported = y[mask]
    global_mean = float(supported.mean()) if supported.size else 0.5
    for kc in range(y.shape[1]):
        keep = mask[:, kc]
        if int(keep.sum()) < 2:
            weights.append(np.zeros(x.shape[1], dtype=np.float64))
            biases.append(float(y[keep, kc].mean()) if bool(keep.any()) else global_mean)
            continue
        local = x[keep].astype(np.float64)
        target = y[keep, kc].astype(np.float64)
        x_mean = local.mean(axis=0)
        y_mean = float(target.mean())
        centered_x = local - x_mean
        centered_y = target - y_mean
        if centered_x.shape[1] <= centered_x.shape[0]:
            coefficient = np.linalg.solve(
                centered_x.T @ centered_x
                + alpha * np.eye(centered_x.shape[1], dtype=np.float64),
                centered_x.T @ centered_y,
            )
        else:
            coefficient = centered_x.T @ np.linalg.solve(
                centered_x @ centered_x.T
                + alpha * np.eye(centered_x.shape[0], dtype=np.float64),
                centered_y,
            )
        weights.append(coefficient)
        biases.append(y_mean - float(x_mean @ coefficient))
    return np.stack(weights, axis=1), np.asarray(biases)


def predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.clip(x @ weights + bias, 0.0, 1.0)


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        raise ValueError("Scoring stratum has no cells")
    return float(np.sqrt(np.square(prediction[mask] - target[mask]).mean()))


def select_and_fit(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
    alphas: list[float],
) -> dict[str, Any]:
    train_normal, mean, scale = normalize_fit(train_x)
    valid_normal = normalize_apply(valid_x, mean, scale)
    selected: tuple[float, float] | None = None
    for alpha in alphas:
        weights, bias = fit_weighted_ridge(
            train_normal, targets["train"][0], targets["train"][1], alpha
        )
        value = masked_rmse(
            predict(valid_normal, weights, bias),
            targets["valid"][0],
            targets["valid"][1],
        )
        key = (value, alpha)
        if selected is None or key < selected:
            selected = key
    assert selected is not None
    fit_x = np.concatenate((train_x, valid_x), axis=0)
    fit_y = np.concatenate((targets["train"][0], targets["valid"][0]), axis=0)
    fit_mask = np.concatenate((targets["train"][1], targets["valid"][1]), axis=0)
    fit_normal, fit_mean, fit_scale = normalize_fit(fit_x)
    weights, bias = fit_weighted_ridge(fit_normal, fit_y, fit_mask, selected[1])
    return {
        "alpha": float(selected[1]),
        "validation_rmse": float(selected[0]),
        "prediction": predict(
            normalize_apply(test_x, fit_mean, fit_scale), weights, bias
        ),
    }


def population_prediction(
    targets: dict[str, tuple[np.ndarray, np.ndarray]], n: int
) -> np.ndarray:
    y = np.concatenate((targets["train"][0], targets["valid"][0]), axis=0)
    mask = np.concatenate((targets["train"][1], targets["valid"][1]), axis=0)
    denominator = mask.sum(axis=0).clip(min=1)
    mean = (y * mask).sum(axis=0) / denominator
    return np.repeat(mean[None, :], n, axis=0)


def visible_support(rows: list[dict[str, Any]]) -> np.ndarray:
    vocabulary = [str(value) for value in rows[0]["kc_vocabulary"]]
    index = {name: position for position, name in enumerate(vocabulary)}
    output = np.zeros((len(rows), len(vocabulary)), dtype=bool)
    for row_index, row in enumerate(rows):
        if [str(value) for value in row["kc_vocabulary"]] != vocabulary:
            raise ValueError("KC vocabulary drift")
        for event in row["state_history"]:
            for name in set(str(value) for value in event["kc_names"]):
                if name in index:
                    output[row_index, index[name]] = True
    return output


def cell_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    delta = prediction[mask] - target[mask]
    return {
        "cells": int(mask.sum()),
        "students_with_cells": int(mask.any(axis=1).sum()),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
    }


def permutation_null(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    observed = masked_rmse(prediction, target, mask)
    identity = np.arange(len(prediction))
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        order = rng.permutation(len(prediction))
        if np.array_equal(order, identity):
            order = np.roll(order, 1)
        null[replicate] = masked_rmse(prediction[order], target, mask)
    return {
        "observed_rmse": observed,
        "permuted_rmse_mean": float(null.mean()),
        "permuted_rmse_ci95": [
            float(value) for value in np.quantile(null, [0.025, 0.975])
        ],
        "delta_permuted_mean_minus_observed": float(null.mean() - observed),
        "p_one_sided": float((1 + np.sum(null <= observed)) / (replicates + 1)),
        "replicates": replicates,
    }


def student_rmse(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    output = np.full(len(prediction), np.nan, dtype=np.float64)
    for index in range(len(prediction)):
        if bool(mask[index].any()):
            output[index] = np.sqrt(
                np.square(prediction[index, mask[index]] - target[index, mask[index]]).mean()
            )
    return output


def paired_bootstrap(
    model: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left = student_rmse(model, target, mask)
    right = student_rmse(reference, target, mask)
    keep = np.isfinite(left) & np.isfinite(right)
    delta = left[keep] - right[keep]
    if not len(delta):
        raise ValueError("Bootstrap stratum has no students")
    rng = np.random.default_rng(seed)
    samples = delta[rng.integers(0, len(delta), size=(replicates, len(delta)))].mean(axis=1)
    return {
        "metric": "mean_student_RMSE_delta_model_minus_reference",
        "students": int(len(delta)),
        "value": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_probability_model_better": float(np.mean(samples < 0)),
        "replicates": replicates,
    }


def validate_artifacts(
    manifest_dir: Path,
    feature_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, Any]],
]:
    rows = {split: read_jsonl(manifest_dir / f"{split}.jsonl") for split in SPLITS}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    metadata = {}
    for split in SPLITS:
        path = feature_dir / f"{split}.npz"
        with np.load(path) as payload:
            arrays[split] = {name: payload[name].copy() for name in payload.files}
        metadata_path = feature_dir / f"{split}.json"
        metadata[split] = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata[split]["example_ids"] != [str(row["example_id"]) for row in rows[split]]:
            raise ValueError(f"{split} feature order differs from manifest")
        if arrays[split]["full"].shape[0] != len(rows[split]):
            raise ValueError(f"{split} feature cardinality differs")
        if metadata[split].get("npz_sha256") != sha256(path):
            raise ValueError(f"{split} feature hash differs")
    checkpoints = {
        (value["adapter_sha256"], value["recurrent_state_sha256"])
        for value in metadata.values()
    }
    if len(checkpoints) != 1:
        raise ValueError("Feature splits come from different checkpoints")
    return rows, arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    rows, arrays, metadata = validate_artifacts(args.manifest_dir, args.feature_dir)
    targets = {
        split: (
            arrays[split]["future_rates"].astype(np.float32),
            arrays[split]["future_mask"].astype(bool),
        )
        for split in SPLITS
    }
    state = select_and_fit(
        arrays["train"]["full"],
        arrays["valid"]["full"],
        arrays["test"]["full"],
        targets,
        args.alphas,
    )
    history = select_and_fit(
        arrays["train"]["history"],
        arrays["valid"]["history"],
        arrays["test"]["history"],
        targets,
        args.alphas,
    )
    population = population_prediction(targets, len(rows["test"]))
    future_mask = targets["test"][1]
    observed = visible_support(rows["test"])
    strata = {
        "all_future_supported": future_mask,
        "seen_in_updater_history": future_mask & observed,
        "unseen_in_updater_history": future_mask & ~observed,
    }
    result_strata = {}
    for index, (name, mask) in enumerate(strata.items()):
        result_strata[name] = {
            "state8": cell_metrics(state["prediction"], targets["test"][0], mask),
            "same_window_kc_outcomes": cell_metrics(
                history["prediction"], targets["test"][0], mask
            ),
            "train_valid_population": cell_metrics(
                population, targets["test"][0], mask
            ),
            "state8_ownership_permutation": permutation_null(
                state["prediction"],
                targets["test"][0],
                mask,
                replicates=args.permutation_replicates,
                seed=args.seed + 10 * index,
            ),
            "state8_vs_same_window": paired_bootstrap(
                state["prediction"],
                history["prediction"],
                targets["test"][0],
                mask,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 10 * index + 1,
            ),
            "state8_vs_population": paired_bootstrap(
                state["prediction"],
                population,
                targets["test"][0],
                mask,
                replicates=args.bootstrap_replicates,
                seed=args.seed + 10 * index + 2,
            ),
        }
    result = {
        "stage": "state8_profile_seen_unseen_feature_audit_v1",
        "status": "complete",
        "information_route": "frozen State8 features only for State8 readout",
        "student_id_visible_to_readout": False,
        "raw_history_visible_to_state8_readout": False,
        "test_labels_used_for_selection": False,
        "same_window_control_uses_exact_updater_visible_history": True,
        "stratification_uses_input_kc_names_not_future_outcomes": True,
        "selection": {
            "state_alpha": state["alpha"],
            "state_validation_rmse": state["validation_rmse"],
            "history_alpha": history["alpha"],
            "history_validation_rmse": history["validation_rmse"],
        },
        "students": {split: len(rows[split]) for split in SPLITS},
        "checkpoint": {
            "adapter_sha256": metadata["test"]["adapter_sha256"],
            "recurrent_state_sha256": metadata["test"]["recurrent_state_sha256"],
        },
        "feature_artifacts": {
            split: {
                "npz_sha256": sha256(args.feature_dir / f"{split}.npz"),
                "metadata_sha256": sha256(args.feature_dir / f"{split}.json"),
            }
            for split in SPLITS
        },
        "strata": result_strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
