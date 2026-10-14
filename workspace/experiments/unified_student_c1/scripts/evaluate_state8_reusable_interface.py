#!/usr/bin/env python3
"""Evaluate State8 as a reusable, readable student-state interface.

The primary C4 claim in this protocol is representation faithfulness, not
literal donor-text transport.  A frozen recurrent state must support a
student-disjoint future-behaviour readout, outperform controls that destroy
state ownership, and remain consistent across two disjoint views of the same
history.  Cross-student swap is deliberately not a scientific gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_dbe_student_profile_probe as probe
import evaluate_dbe_student_profile_transport_v3 as profile_v3
from evaluate_recurrent_formal_protocols import load_wrapper, pad_prompt_batches
from recurrent_student_model import build_update_text
from train_recurrent_student_diagnostic import alias_text, encode_prompt, move


STAGE = "state8_reusable_student_interface_v1"
FEATURE_VIEW = "layerwise_mean_slots_rp"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    return probe.random_projection_matrix(input_dim, output_dim, seed)


def canonical_features(
    state: torch.Tensor, projection_dim: int, projection_seed: int
) -> np.ndarray:
    """Compress [B,L,S,D] without collapsing the layerwise trajectory."""
    value = state.detach().float().cpu().numpy()
    if value.ndim != 4:
        raise ValueError(f"State8 trajectory must be [B,L,S,D], got {value.shape}")
    per_layer = value.mean(axis=2)
    projection = stable_projection(per_layer.shape[-1], projection_dim, projection_seed)
    features = np.einsum("nld,dp->nlp", per_layer, projection, optimize=True)
    return features.reshape(features.shape[0], -1).astype(np.float32)


def update_texts(row: dict[str, Any], view: str) -> list[str]:
    events = list(row["state_history"])
    if view == "odd":
        events = events[::2]
    elif view == "even":
        events = events[1::2]
    elif view != "full":
        raise ValueError(f"Unknown history view: {view}")
    if not events:
        raise ValueError(f"Empty {view} history for {row['example_id']}")
    return [build_update_text(event) for event in events]


@torch.inference_mode()
def extract_features(
    wrapper,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    view: str,
    max_length: int,
    batch_size: int,
    projection_dim: int,
    projection_seed: int,
) -> tuple[np.ndarray, list[int]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    device = next(wrapper.parameters()).device
    alias = wrapper.state_token_rendered_alias
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    updates = [update_texts(row, view) for row in rows]
    by_depth: dict[int, list[int]] = {}
    for index, values in enumerate(updates):
        by_depth.setdefault(len(values), []).append(index)
    output: list[np.ndarray | None] = [None] * len(rows)
    state_shape: list[int] | None = None
    completed = 0
    for depth, indices in sorted(by_depth.items()):
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            state = wrapper.initial_state(len(selected), device=device)
            for position in range(depth):
                encoded = [
                    move(
                        encode_prompt(
                            tokenizer,
                            alias_text(updates[row_index][position], alias),
                            max_length,
                        ),
                        device,
                    )
                    for row_index in selected
                ]
                batch = pad_prompt_batches(encoded, int(pad_id))
                state = wrapper.update_state(state, **batch).state
            if state is None:
                raise AssertionError("Observed history produced no recurrent state")
            observed_shape = list(state.shape[1:])
            if state_shape is None:
                state_shape = observed_shape
            elif state_shape != observed_shape:
                raise ValueError("Recurrent state shape changed across batches")
            values = canonical_features(state, projection_dim, projection_seed)
            for row_index, value in zip(selected, values, strict=True):
                output[row_index] = value
            completed += len(selected)
            print(
                json.dumps(
                    {"view": view, "states": completed, "total": len(rows)},
                    sort_keys=True,
                ),
                flush=True,
            )
            del state, batch, encoded, values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if state_shape is None or any(value is None for value in output):
        raise AssertionError("Feature extraction was incomplete")
    return np.stack([value for value in output if value is not None]), state_shape


def history_features(rows: list[dict[str, Any]], vocabulary: list[str]) -> np.ndarray:
    return profile_v3.history_statistics(rows, vocabulary, "state_history")


def future_targets(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return probe.future_matrices(rows)


def save_split(
    output: Path,
    rows: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    state_shape: list[int],
    vocabulary: list[str],
    args: argparse.Namespace,
) -> None:
    rates, mask, counts = future_targets(rows)
    arrays: dict[str, np.ndarray] = {
        "full": features["full"].astype(np.float16),
        "history": history_features(rows, vocabulary).astype(np.float32),
        "future_rates": rates.astype(np.float32),
        "future_mask": mask.astype(np.uint8),
        "future_counts": counts.astype(np.float32),
    }
    for view in ("odd", "even"):
        if view in features:
            arrays[view] = features[view].astype(np.float16)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {
        "stage": STAGE,
        "status": "complete",
        "split": args.split,
        "rows": len(rows),
        "example_ids": [str(row["example_id"]) for row in rows],
        "student_ids": [str(row["student_id"]) for row in rows],
        "feature_view": FEATURE_VIEW,
        "feature_width": int(features["full"].shape[1]),
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "state_shape": state_shape,
        "state_token_count": 8,
        "state_token_alias": args.state_token_alias,
        "information_route": "chronological_updates_then_frozen_state8",
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "chronological_future_targets": True,
        "odd_even_views": sorted(set(features) & {"odd", "even"}),
        "manifest_sha256": sha256(args.rows_jsonl),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "recurrent_state_sha256": sha256(args.recurrent_state),
        "npz_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return probe.normalize_fit(x.astype(np.float32))


def fit_readout(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: list[float],
) -> dict[str, Any]:
    train_normal, mean, scale = normalize_fit(train_x)
    valid_normal = probe.normalize_apply(valid_x, mean, scale)
    alpha, _weights, _bias = probe.choose_alpha(
        train_normal,
        targets["train"][0],
        targets["train"][1],
        valid_normal,
        targets["valid"][0],
        targets["valid"][1],
        alphas,
    )
    fit_x = np.concatenate((train_x, valid_x), axis=0).astype(np.float32)
    fit_normal, fit_mean, fit_scale = normalize_fit(fit_x)
    fit_y = np.concatenate((targets["train"][0], targets["valid"][0]), axis=0)
    fit_mask = np.concatenate((targets["train"][1], targets["valid"][1]), axis=0)
    weights, bias = probe.fit_weighted_ridge(fit_normal, fit_y, fit_mask, alpha)
    prediction = probe.predict_ridge(
        probe.normalize_apply(test_x.astype(np.float32), fit_mean, fit_scale),
        weights,
        bias,
    )
    return {
        "alpha": float(alpha),
        "mean": fit_mean,
        "scale": fit_scale,
        "weights": weights,
        "bias": bias,
        "prediction": prediction,
    }


def apply_readout(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    normalized = probe.normalize_apply(
        x.astype(np.float32), model["mean"], model["scale"]
    )
    return probe.predict_ridge(normalized, model["weights"], model["bias"])


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    weakness_candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    # ``mask`` is the post-cutoff observation mask and is appropriate for
    # scoring future-rate regression.  It is *not* an inference-time input:
    # using a student's future support to choose which KCs may be named as a
    # weakness leaks part of the held-out trajectory.  The primary readable
    # profile therefore uses one candidate set fixed from train/validation
    # support before test labels are inspected.
    if weakness_candidate_mask is None:
        weakness_candidate_mask = mask
        candidate_protocol = "legacy_per_student_future_support_diagnostic"
    else:
        weakness_candidate_mask = np.asarray(weakness_candidate_mask, dtype=bool)
        if weakness_candidate_mask.ndim != 1 or weakness_candidate_mask.shape[0] != target.shape[1]:
            raise ValueError("Weakness candidate mask must be one fixed [KC] vector")
        if int(weakness_candidate_mask.sum()) < 2:
            raise ValueError("Weakness candidate mask needs at least two KCs")
        weakness_candidate_mask = np.broadcast_to(
            weakness_candidate_mask, (len(rows), weakness_candidate_mask.shape[0])
        )
        candidate_protocol = "fixed_train_valid_supported_kcs"
    return {
        "future_rate": probe.masked_metrics(prediction, target, mask),
        "weakness": {
            **probe.weakness_metrics(prediction, rows, weakness_candidate_mask),
            "candidate_protocol": candidate_protocol,
        },
    }


def prediction_stability_null(
    left: np.ndarray,
    right: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Stability predictions must be aligned [student, KC]")
    observed = float(np.square(left - right).mean())
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [
            np.square(left - right[rng.permutation(len(right))]).mean()
            for _ in range(replicates)
        ],
        dtype=np.float64,
    )
    return {
        "metric": "mean_squared_difference_between_KC_readouts",
        "same_student": observed,
        "permuted_student_mean": float(null.mean()),
        "permuted_student_ci95": [
            float(value) for value in np.quantile(null, [0.025, 0.975])
        ],
        "p_one_sided": float((1 + np.sum(null <= observed)) / (replicates + 1)),
        "replicates": replicates,
    }


def representation_stability_null(
    left: np.ndarray,
    right: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    left = left / np.linalg.norm(left, axis=1, keepdims=True).clip(min=1e-12)
    right = right / np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-12)
    observed = float(np.sum(left * right, axis=1).mean())
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [
            np.sum(left * right[rng.permutation(len(right))], axis=1).mean()
            for _ in range(replicates)
        ],
        dtype=np.float64,
    )
    return {
        "metric": "cosine_similarity_of_disjoint_history_states",
        "same_student": observed,
        "permuted_student_mean": float(null.mean()),
        "permuted_student_ci95": [
            float(value) for value in np.quantile(null, [0.025, 0.975])
        ],
        "p_one_sided": float((1 + np.sum(null >= observed)) / (replicates + 1)),
        "replicates": replicates,
    }


def prediction_reliability(
    left: np.ndarray,
    right: np.ndarray,
    fit_target_variance: np.ndarray,
) -> dict[str, Any]:
    """Report noise-adjusted agreement without turning collapse into a pass.

    Odd and even histories are complementary rather than replicate
    measurements, so this remains a descriptive diagnostic.  The denominator
    is frozen from fitting-split outcome variance; a constant prediction has
    zero between-student variance and therefore cannot appear reliable merely
    because both views collapse to the same vector.
    """
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Reliability predictions must be aligned [student, KC]")
    variance = np.asarray(fit_target_variance, dtype=np.float64)
    if variance.ndim != 1 or variance.shape[0] != left.shape[1]:
        raise ValueError("Fitting variance must have one value per KC")
    if not bool(np.isfinite(left).all()) or not bool(np.isfinite(right).all()):
        raise ValueError("Reliability predictions must be finite")
    if not bool(np.isfinite(variance).all()) or bool((variance < 0).any()):
        raise ValueError("Fitting variance must be finite and non-negative")
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    within = np.square(left - right).mean(axis=0) / 2.0
    between = np.var((left + right) / 2.0, axis=0, ddof=1)
    signal = np.maximum(0.0, between - within)
    supported = variance > 1e-10
    scale = np.maximum(variance, 1e-12)
    return {
        "metric": "noise_adjusted_disjoint_view_prediction_reliability",
        "role": "diagnostic_only_not_a_scientific_gate",
        "supported_kcs": int(supported.sum()),
        "mean_within_student_noise_over_fit_variance": float(
            np.mean(within[supported] / scale[supported])
        ) if bool(supported.any()) else None,
        "mean_between_student_variance_over_fit_variance": float(
            np.mean(between[supported] / scale[supported])
        ) if bool(supported.any()) else None,
        "mean_noise_adjusted_signal_over_fit_variance": float(
            np.mean(signal[supported] / scale[supported])
        ) if bool(supported.any()) else None,
    }


def scientific_gates(result: dict[str, Any]) -> dict[str, bool]:
    uncertainty = result["uncertainty"]
    state_vs_population = uncertainty["state8_vs_population"]
    state_vs_history = uncertainty["state8_vs_same_window_kc_outcomes"]
    ownership = uncertainty["state_ownership_permutation_null"]
    stability = uncertainty["disjoint_history_prediction_stability"]
    return {
        "heldout_student_support": result["students"]["test"] >= 80,
        "predictive_utility_rmse": (
            state_vs_population["rmse_delta_model_minus_reference"]
            ["bootstrap_probability_model_better"]
            > 0.95
        ),
        "readable_weakness_utility": (
            state_vs_population["weak_f1_delta_model_minus_reference"]
            ["bootstrap_probability_model_better"]
            > 0.95
        ),
        # Population is a necessary null but far too weak to establish that
        # State8 carries educational information beyond the correctness/count
        # statistics available from exactly the same observed window.
        "predictive_utility_beyond_same_window_rmse": (
            state_vs_history["rmse_delta_model_minus_reference"]
            ["bootstrap_probability_model_better"]
            > 0.95
        ),
        "readable_weakness_beyond_same_window": (
            state_vs_history["weak_f1_delta_model_minus_reference"]
            ["bootstrap_probability_model_better"]
            > 0.95
        ),
        "state_ownership_matters": ownership["rmse"]["p_one_sided"] < 0.05,
        "same_student_disjoint_history_stability": stability["p_one_sided"] < 0.05,
        "closed_information_route": (
            result["raw_history_visible_to_readout"] is False
            and result["student_id_visible_to_readout"] is False
        ),
    }


def readable_profiles(
    prediction: np.ndarray,
    rows: list[dict[str, Any]],
    vocabulary: list[str],
    candidate_mask: np.ndarray,
) -> list[dict[str, Any]]:
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.ndim != 1 or candidate_mask.shape[0] != len(vocabulary):
        raise ValueError("Readable-profile candidate mask must be one fixed [KC] vector")
    eligible = np.flatnonzero(candidate_mask)
    if len(eligible) < 2:
        raise ValueError("Readable profiles require at least two fixed candidate KCs")
    records = []
    for values, row in zip(prediction, rows, strict=True):
        ordered = eligible[np.argsort(values[eligible], kind="stable")]
        weak = [vocabulary[int(index)] for index in ordered[:2]]
        strong = [vocabulary[int(index)] for index in ordered[-2:][::-1]]
        records.append(
            {
                "example_id": row["example_id"],
                "predicted_weaknesses": weak,
                "predicted_strengths": strong,
                "gold_future_weaknesses": list(row["weak_kcs"]),
                "profile": (
                    f"The student is predicted to need support with {', '.join(weak)} "
                    f"and to be relatively strong in {', '.join(strong)}."
                ),
                "raw_history_visible_to_readout": False,
                "student_id_visible_to_readout": False,
                "future_support_visible_to_readout": False,
                "candidate_protocol": "fixed_train_valid_supported_kcs",
            }
        )
    return records


def analyze(args: argparse.Namespace) -> None:
    raw = {
        split: read_jsonl(args.manifest_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    vocabulary = profile_v3.validate_rows(raw)
    payload = {
        split: np.load(args.feature_dir / f"{split}.npz")
        for split in ("train", "valid", "test")
    }
    metadata = {
        split: json.loads(
            (args.feature_dir / f"{split}.json").read_text(encoding="utf-8")
        )
        for split in ("train", "valid", "test")
    }
    for split in payload:
        if payload[split]["full"].shape[0] != len(raw[split]):
            raise ValueError(f"Feature cardinality mismatch for {split}")
        if metadata[split]["example_ids"] != [
            str(row["example_id"]) for row in raw[split]
        ]:
            raise ValueError(f"Feature order mismatch for {split}")
    if "odd" not in payload["test"] or "even" not in payload["test"]:
        raise ValueError("Test features require odd/even disjoint-history views")

    targets = {
        split: (
            payload[split]["future_rates"].astype(np.float32),
            payload[split]["future_mask"].astype(bool),
            payload[split]["future_counts"].astype(np.float32),
        )
        for split in payload
    }
    alphas = [float(value) for value in args.alphas]
    state_model = fit_readout(
        payload["train"]["full"],
        payload["valid"]["full"],
        payload["test"]["full"],
        targets,
        alphas,
    )
    history_model = fit_readout(
        payload["train"]["history"],
        payload["valid"]["history"],
        payload["test"]["history"],
        targets,
        alphas,
    )
    state_prediction = state_model["prediction"]
    history_prediction = history_model["prediction"]
    fit_y = np.concatenate((targets["train"][0], targets["valid"][0]), axis=0)
    fit_mask = np.concatenate((targets["train"][1], targets["valid"][1]), axis=0)
    fit_target_variance = np.asarray(
        [
            float(np.var(fit_y[fit_mask[:, index], index], ddof=1))
            if int(fit_mask[:, index].sum()) >= 2
            else 0.0
            for index in range(fit_y.shape[1])
        ],
        dtype=np.float64,
    )
    # Freeze the set of KCs that a readable profile may name using fitting
    # splits only.  The per-test-student future support remains a scoring mask
    # for rate RMSE, never a prediction-time candidate list.
    weakness_candidate_mask = fit_mask.sum(axis=0) >= 2
    if int(weakness_candidate_mask.sum()) < 2:
        raise ValueError("Fitting splits support fewer than two profile KCs")
    population_prediction = probe.mean_predictor(
        fit_y, fit_mask, len(raw["test"])
    )
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(state_prediction))
    if np.all(permutation == np.arange(len(permutation))):
        permutation = np.roll(permutation, 1)
    permuted_prediction = state_prediction[permutation]
    mean_state = np.repeat(
        np.concatenate(
            (payload["train"]["full"], payload["valid"]["full"]), axis=0
        ).mean(axis=0, keepdims=True),
        len(raw["test"]),
        axis=0,
    )
    mean_state_prediction = apply_readout(state_model, mean_state)
    target, mask, _counts = targets["test"]
    metric_values = {
        "state8": metrics(
            state_prediction,
            target,
            mask,
            raw["test"],
            weakness_candidate_mask=weakness_candidate_mask,
        ),
        "same_window_kc_outcomes": metrics(
            history_prediction,
            target,
            mask,
            raw["test"],
            weakness_candidate_mask=weakness_candidate_mask,
        ),
        "train_valid_population": metrics(
            population_prediction,
            target,
            mask,
            raw["test"],
            weakness_candidate_mask=weakness_candidate_mask,
        ),
        "permuted_state_ownership": metrics(
            permuted_prediction,
            target,
            mask,
            raw["test"],
            weakness_candidate_mask=weakness_candidate_mask,
        ),
        "mean_training_state": metrics(
            mean_state_prediction,
            target,
            mask,
            raw["test"],
            weakness_candidate_mask=weakness_candidate_mask,
        ),
    }
    odd_prediction = apply_readout(state_model, payload["test"]["odd"])
    even_prediction = apply_readout(state_model, payload["test"]["even"])
    uncertainty = {
        "state8_vs_population": profile_v3.paired_student_bootstrap(
            state_prediction,
            population_prediction,
            target,
            mask,
            raw["test"],
            args.bootstrap_replicates,
            args.seed + 1,
            weakness_candidate_mask,
        ),
        "state8_vs_same_window_kc_outcomes": profile_v3.paired_student_bootstrap(
            state_prediction,
            history_prediction,
            target,
            mask,
            raw["test"],
            args.bootstrap_replicates,
            args.seed + 2,
            weakness_candidate_mask,
        ),
        "state_ownership_permutation_null": profile_v3.permutation_null(
            state_prediction,
            target,
            mask,
            raw["test"],
            args.permutation_replicates,
            args.seed + 3,
            weakness_candidate_mask,
        ),
        "disjoint_history_prediction_stability": prediction_stability_null(
            odd_prediction,
            even_prediction,
            args.permutation_replicates,
            args.seed + 4,
        ),
        "disjoint_history_prediction_reliability_diagnostic": prediction_reliability(
            odd_prediction,
            even_prediction,
            fit_target_variance,
        ),
        "disjoint_history_representation_stability_diagnostic": (
            representation_stability_null(
                payload["test"]["odd"],
                payload["test"]["even"],
                args.permutation_replicates,
                args.seed + 5,
            )
        ),
    }
    result: dict[str, Any] = {
        "stage": STAGE,
        "status": "complete",
        "claim": (
            "State8 is a stable, readable, reusable student representation whose "
            "ownership matters for held-out educational predictions."
        ),
        "excluded_primary_claim": (
            "Cross-student state swaps must force literal donor diagnosis text."
        ),
        "swap_role": "diagnostic_only_not_a_scientific_gate",
        "feature_view": FEATURE_VIEW,
        "state_shape": metadata["test"]["state_shape"],
        "state_token_count": 8,
        "students": {split: len(raw[split]) for split in raw},
        "student_disjoint": True,
        "chronological_future_targets": True,
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "future_support_visible_to_readout": False,
        "weakness_candidate_protocol": "fixed_train_valid_supported_kcs",
        "weakness_candidate_kc_indices": [
            int(index) for index in np.flatnonzero(weakness_candidate_mask)
        ],
        "readout_selection": {
            "state_alpha": state_model["alpha"],
            "history_alpha": history_model["alpha"],
            "selection_split": "validation_students_only",
            "test_labels_used_for_selection": False,
        },
        "metrics": metric_values,
        "uncertainty": uncertainty,
        "feature_artifacts": {
            split: {
                "npz_sha256": sha256(args.feature_dir / f"{split}.npz"),
                "metadata_sha256": sha256(args.feature_dir / f"{split}.json"),
                "checkpoint": {
                    "adapter": metadata[split]["adapter_sha256"],
                    "recurrent_state": metadata[split]["recurrent_state_sha256"],
                },
            }
            for split in payload
        },
    }
    checkpoint_hashes = {
        (
            value["checkpoint"]["adapter"],
            value["checkpoint"]["recurrent_state"],
        )
        for value in result["feature_artifacts"].values()
    }
    if len(checkpoint_hashes) != 1:
        raise ValueError("Feature splits were extracted from different checkpoints")
    result["gates"] = scientific_gates(result)
    result["scientific_gate_pass"] = all(result["gates"].values())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "readable_profiles.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in readable_profiles(
            state_prediction,
            raw["test"],
            vocabulary,
            weakness_candidate_mask,
        ):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


def extract(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.rows_jsonl)
    tokenizer, wrapper = load_wrapper(
        args.model,
        str(args.adapter),
        args.recurrent_state,
        args.state_token_alias,
        "layerwise",
    )
    if wrapper.state_token_count != 8:
        raise ValueError("Reusable-interface evaluation requires State8")
    features = {}
    features["full"], state_shape = extract_features(
        wrapper,
        tokenizer,
        rows,
        view="full",
        max_length=args.max_length,
        batch_size=args.batch_size,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
    )
    if args.split == "test":
        for view in ("odd", "even"):
            features[view], split_shape = extract_features(
                wrapper,
                tokenizer,
                rows,
                view=view,
                max_length=args.max_length,
                batch_size=args.batch_size,
                projection_dim=args.projection_dim,
                projection_seed=args.projection_seed,
            )
            if split_shape != state_shape:
                raise ValueError("Disjoint history views changed the State8 shape")
    vocabulary = [str(value) for value in rows[0]["kc_vocabulary"]]
    save_split(args.output, rows, features, state_shape, vocabulary, args)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--model", required=True)
    extract_parser.add_argument("--adapter", type=Path, required=True)
    extract_parser.add_argument("--recurrent-state", type=Path, required=True)
    extract_parser.add_argument("--rows-jsonl", type=Path, required=True)
    extract_parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    extract_parser.add_argument("--max-length", type=int, default=1536)
    extract_parser.add_argument("--batch-size", type=int, default=4)
    extract_parser.add_argument("--projection-dim", type=int, default=64)
    extract_parser.add_argument("--projection-seed", type=int, default=20260818)

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--manifest-dir", type=Path, required=True)
    analyze_parser.add_argument("--feature-dir", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0]
    )
    analyze_parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    analyze_parser.add_argument("--permutation-replicates", type=int, default=2000)
    analyze_parser.add_argument("--seed", type=int, default=20260818)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "extract":
        extract(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
