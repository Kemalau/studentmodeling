#!/usr/bin/env python3
"""Evaluate whether frozen State8 trajectories contain reusable profiles.

This is deliberately a lightweight readout, not an LLM self-explanation test.
It extracts one frozen state per student, projects several views of the
64x8x5120 trajectory, and fits ridge heads to future KC correctness.  Swapping
the input state is therefore an exact functional test: the readout has no
student ID, history, or query channel through which to bypass the state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_recurrent_formal_protocols import load_wrapper, pad_prompt_batches
from train_recurrent_student_diagnostic import alias_text, encode_prompt, move


STAGE = "frozen_state8_future_profile_probe_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def extract_direct_features(
    wrapper,
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    projection_dim: int,
    projection_seed: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Encode history once, then retain only the State8 residual trajectory."""
    if not rows:
        raise ValueError("Profile extraction requires rows")
    device = wrapper.backbone.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    outputs: dict[str, list[np.ndarray]] = {}
    state_shape: list[int] | None = None
    for start in range(0, len(rows), batch_size):
        selected = rows[start : start + batch_size]
        encoded = [
            move(
                encode_prompt(
                    tokenizer,
                    alias_text(row["prompt"], wrapper.state_token_rendered_alias),
                    max_length,
                ),
                device,
            )
            for row in selected
        ]
        batch = pad_prompt_batches(encoded, int(pad_id))
        state = wrapper.capture_direct_state(**batch).state
        state_numpy = state.detach().float().cpu().numpy()
        if state_shape is None:
            state_shape = list(state_numpy.shape[1:])
        elif state_shape != list(state_numpy.shape[1:]):
            raise ValueError("Direct-state shape changed across batches")
        for name, values in feature_views(
            state_numpy, projection_dim, projection_seed
        ).items():
            outputs.setdefault(name, []).append(values)
        del state, state_numpy
        print(
            json.dumps({"states": min(start + batch_size, len(rows)), "total": len(rows)}),
            flush=True,
        )
    if state_shape is None:
        raise AssertionError("No state shape was recorded")
    return {
        name: np.concatenate(parts, axis=0) for name, parts in outputs.items()
    }, state_shape


def random_projection_matrix(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Achlioptas projection: reproducible and cheap to store/recompute.
    values = rng.choice(
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        size=(input_dim, output_dim),
        p=(1 / 6, 2 / 3, 1 / 6),
    )
    return values * math.sqrt(3.0 / output_dim)


def feature_views(
    state: np.ndarray,
    projection_dim: int,
    projection_seed: int,
) -> dict[str, np.ndarray]:
    if state.ndim != 4:
        raise ValueError("State trajectory must be [N,L,S,D]")
    views = {
        "final_layer_mean_slots": state[:, -1].mean(axis=1),
        "mean_layers_mean_slots": state.mean(axis=(1, 2)),
    }
    per_layer = state.mean(axis=2)
    projection = random_projection_matrix(
        per_layer.shape[-1], projection_dim, projection_seed
    )
    views["layerwise_mean_slots_rp"] = np.einsum(
        "nld,dp->nlp", per_layer, projection, optimize=True
    ).reshape(per_layer.shape[0], -1)
    return {name: values.astype(np.float32) for name, values in views.items()}


def future_matrices(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray([row["future_counts"] for row in rows], dtype=np.float32)
    correct = np.asarray([row["future_correct"] for row in rows], dtype=np.float32)
    mask = np.asarray([row["future_eligible_mask"] for row in rows], dtype=bool)
    rates = (correct + 1.0) / (counts + 2.0)
    return rates, mask, counts


def prefix_rate_features(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = np.asarray([row["prefix_counts"] for row in rows], dtype=np.float32)
    correct = np.asarray([row["prefix_correct"] for row in rows], dtype=np.float32)
    return (correct + 1.0) / (counts + 2.0)


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (x - mean) / scale, mean, scale


def normalize_apply(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / scale


def fit_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one ridge head per KC using only supported post-cutoff targets."""
    weights = []
    biases = []
    supported_values = y[mask]
    global_mean = float(supported_values.mean()) if supported_values.size else 0.5
    for kc in range(y.shape[1]):
        keep = mask[:, kc]
        if keep.sum() < 2:
            # Some prefix-frequent KCs have no post-cutoff observations in a
            # student-disjoint split.  They cannot support a learned slope.
            # Retain the frozen output schema with a train-only constant so a
            # missing target never crashes or leaks validation/test outcomes.
            weights.append(np.zeros(x.shape[1], dtype=np.float64))
            biases.append(float(y[keep, kc].mean()) if keep.any() else global_mean)
            continue
        local = x[keep].astype(np.float64)
        target = y[keep, kc].astype(np.float64)
        x_mean = local.mean(axis=0)
        y_mean = target.mean()
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
        biases.append(y_mean - x_mean @ coefficient)
    return np.stack(weights, axis=1), np.asarray(biases)


def predict_ridge(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.clip(x @ weights + bias, 0.0, 1.0)


def masked_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    delta = prediction[mask] - target[mask]
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "supported_student_kc_cells": int(mask.sum()),
    }


def weak_set(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in row["weak_kc_indices"]}


def top2_weak(prediction: np.ndarray, eligible: np.ndarray) -> set[int]:
    indices = np.flatnonzero(eligible)
    if len(indices) < 2:
        raise ValueError("Weakness evaluation needs two eligible KCs")
    order = indices[np.argsort(prediction[indices], kind="stable")]
    return {int(value) for value in order[:2]}


def weakness_metrics(
    prediction: np.ndarray, rows: list[dict[str, Any]], mask: np.ndarray
) -> dict[str, float]:
    f1 = []
    exact = []
    hit = []
    for pred, row, eligible in zip(prediction, rows, mask, strict=True):
        gold = weak_set(row)
        guessed = top2_weak(pred, eligible)
        overlap = len(gold.intersection(guessed))
        # Both sets contain two labels, hence precision=recall=F1=overlap/2.
        f1.append(overlap / 2.0)
        exact.append(float(gold == guessed))
        hit.append(float(overlap > 0))
    return {
        "weak_top2_macro_f1": float(np.mean(f1)),
        "weak_top2_exact_set_accuracy": float(np.mean(exact)),
        "weak_top2_hit_rate": float(np.mean(hit)),
        "students": len(rows),
    }


def mean_predictor(train_y: np.ndarray, train_mask: np.ndarray, n: int) -> np.ndarray:
    denominator = train_mask.sum(axis=0).clip(min=1)
    mean = (train_y * train_mask).sum(axis=0) / denominator
    return np.repeat(mean[None, :], n, axis=0)


def choose_alpha(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    valid_mask: np.ndarray,
    alphas: list[float],
) -> tuple[float, np.ndarray, np.ndarray]:
    selected = None
    for alpha in alphas:
        weights, bias = fit_weighted_ridge(train_x, train_y, train_mask, alpha)
        prediction = predict_ridge(valid_x, weights, bias)
        rmse = masked_metrics(prediction, valid_y, valid_mask)["rmse"]
        key = (rmse, alpha)
        if selected is None or key < selected[0]:
            selected = (key, alpha, weights, bias)
    assert selected is not None
    return selected[1], selected[2], selected[3]


def matched_pairs(rows: list[dict[str, Any]], seed: int) -> list[tuple[int, int]]:
    """Pair students with disjoint weak sets and comparable future support."""
    candidates = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if weak_set(rows[left]).intersection(weak_set(rows[right])):
                continue
            support_gap = abs(
                sum(rows[left]["future_eligible_mask"])
                - sum(rows[right]["future_eligible_mask"])
            )
            candidates.append((support_gap, left, right))
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates.sort(key=lambda value: value[0])
    used = set()
    pairs = []
    for _gap, left, right in candidates:
        if left in used or right in used:
            continue
        used.update((left, right))
        pairs.append((left, right))
    return pairs


def swap_metrics(prediction: np.ndarray, rows: list[dict[str, Any]], seed: int) -> dict[str, float]:
    pairs = matched_pairs(rows, seed)
    if not pairs:
        raise ValueError("Could not construct profile-swap pairs")
    donor_f1 = []
    native_f1 = []
    donor_exact = []
    native_exact = []
    for left, right in pairs:
        for recipient, donor in ((left, right), (right, left)):
            eligible = np.asarray(rows[donor]["future_eligible_mask"], dtype=bool)
            guessed = top2_weak(prediction[donor], eligible)
            donor_gold = weak_set(rows[donor])
            native_gold = weak_set(rows[recipient])
            donor_overlap = len(guessed.intersection(donor_gold)) / 2.0
            native_overlap = len(guessed.intersection(native_gold)) / 2.0
            donor_f1.append(donor_overlap)
            native_f1.append(native_overlap)
            donor_exact.append(float(guessed == donor_gold))
            native_exact.append(float(guessed == native_gold))
    return {
        "pairs": len(pairs),
        "directions": 2 * len(pairs),
        "swap_to_donor_weak_top2_f1": float(np.mean(donor_f1)),
        "swap_to_native_weak_top2_f1": float(np.mean(native_f1)),
        "D_profile_specific": float(np.mean(donor_f1) - np.mean(native_f1)),
        "swap_to_donor_exact_set_accuracy": float(np.mean(donor_exact)),
        "swap_to_native_exact_set_accuracy": float(np.mean(native_exact)),
    }


def evaluate_view(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
    rows: dict[str, list[dict[str, Any]]],
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: list[float],
    seed: int,
) -> dict[str, Any]:
    train_x, mean, scale = normalize_fit(train_x)
    valid_x = normalize_apply(valid_x, mean, scale)
    test_x = normalize_apply(test_x, mean, scale)
    train_y, train_mask, _ = targets["train"]
    valid_y, valid_mask, _ = targets["valid"]
    test_y, test_mask, _ = targets["test"]
    alpha, _weights, _bias = choose_alpha(
        train_x, train_y, train_mask, valid_x, valid_y, valid_mask, alphas
    )
    # Refit on train+valid after selecting alpha; student-disjoint test stays untouched.
    fit_x = np.concatenate((train_x, valid_x), axis=0)
    fit_y = np.concatenate((train_y, valid_y), axis=0)
    fit_mask = np.concatenate((train_mask, valid_mask), axis=0)
    weights, bias = fit_weighted_ridge(fit_x, fit_y, fit_mask, alpha)
    prediction = predict_ridge(test_x, weights, bias)
    return {
        "selected_alpha": alpha,
        "feature_dim": int(test_x.shape[1]),
        "future_rate": masked_metrics(prediction, test_y, test_mask),
        "weakness": weakness_metrics(prediction, rows["test"], test_mask),
        "swap": swap_metrics(prediction, rows["test"], seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--state-batch-size", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    args = parser.parse_args()

    rows = {
        split: read_jsonl(args.manifest_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    vocabulary = rows["train"][0]["kc_vocabulary"]
    if any(row["kc_vocabulary"] != vocabulary for split in rows for row in rows[split]):
        raise ValueError("Profile rows do not share one frozen KC vocabulary")
    tokenizer, wrapper = load_wrapper(
        args.model,
        args.adapter,
        args.recurrent_state,
        args.state_token_alias,
        "layerwise",
    )
    if wrapper.state_token_count != 8:
        raise ValueError("This pilot requires the frozen eight-token state")
    features: dict[str, dict[str, np.ndarray]] = {}
    state_shape: list[int] | None = None
    for split in ("train", "valid", "test"):
        features[split], observed_shape = extract_direct_features(
            wrapper,
            tokenizer,
            rows[split],
            args.max_length,
            args.state_batch_size,
            args.projection_dim,
            args.seed,
        )
        if state_shape is None:
            state_shape = observed_shape
        elif state_shape != observed_shape:
            raise ValueError("State shape differs by data split")
    targets = {split: future_matrices(rows[split]) for split in rows}
    result: dict[str, Any] = {
        "stage": STAGE,
        "status": "complete",
        "source": "DBE-KT22",
        "students": {split: len(rows[split]) for split in rows},
        "student_disjoint": True,
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "persistent_history_kv_cache": False,
        "state_construction": "one_shot_history_to_state8_direct_encoder",
        "state_shape": state_shape,
        "state_token_count": wrapper.state_token_count,
        "kc_vocabulary": vocabulary,
        "manifest": {
            split: sha256(args.manifest_dir / f"{split}.jsonl") for split in rows
        },
        "artifact_sha256": {
            "adapter_model": sha256(Path(args.adapter) / "adapter_model.safetensors"),
            "recurrent_state": sha256(args.recurrent_state),
        },
        "views": {},
    }
    for name in features["train"]:
        result["views"][name] = evaluate_view(
            features["train"][name],
            features["valid"][name],
            features["test"][name],
            rows,
            targets,
            args.alphas,
            args.seed,
        )

    # Baselines receive no frozen state. The prefix baseline is deliberately
    # strong: it exposes the same pre-cutoff KC outcomes used by the encoder.
    prefix = {split: prefix_rate_features(rows[split]) for split in rows}
    result["baselines"] = {
        "prefix_kc_beta_rates": evaluate_view(
            prefix["train"],
            prefix["valid"],
            prefix["test"],
            rows,
            targets,
            args.alphas,
            args.seed,
        )
    }
    train_y, train_mask, _ = targets["train"]
    test_y, test_mask, _ = targets["test"]
    popularity = mean_predictor(train_y, train_mask, len(rows["test"]))
    result["baselines"]["train_population_mean"] = {
        "future_rate": masked_metrics(popularity, test_y, test_mask),
        "weakness": weakness_metrics(popularity, rows["test"], test_mask),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Only compressed views are retained; raw 64x8x5120 trajectories are not
    # persisted, which keeps the artifact small and avoids accidental ID joins.
    np.savez_compressed(
        args.output_dir / "test_features.npz",
        **{name: values for name, values in features["test"].items()},
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
