#!/usr/bin/env python3
"""Window-matched, uncertainty-audited State8 profile evaluation.

V2 compared a State8 built from the final 12 pre-cutoff interactions against a
baseline that summarized the *entire* pre-cutoff prefix.  V3 separates that
extra-information oracle from a same-window KC-outcome baseline, preserves
slot/depth structure in additional frozen-state views, selects each KC readout
on validation students only, and reports paired bootstrap and permutation-null
uncertainty.  The baseline is matched in interaction window, not representation
capacity: State8 also receives question and response text from those events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import evaluate_dbe_student_profile_probe as probe
    import evaluate_dbe_student_profile_transport as v2
    from evaluate_recurrent_formal_protocols import load_wrapper, pad_prompt_batches
    from train_recurrent_student_diagnostic import alias_text, encode_prompt, move
except ModuleNotFoundError:  # Package-style imports in local tests.
    from scripts import evaluate_dbe_student_profile_probe as probe
    from scripts import evaluate_dbe_student_profile_transport as v2
    from scripts.evaluate_recurrent_formal_protocols import (
        load_wrapper,
        pad_prompt_batches,
    )
    from scripts.train_recurrent_student_diagnostic import (
        alias_text,
        encode_prompt,
        move,
    )


STAGE = "frozen_state8_future_profile_transport_v3"


def validate_rows(rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Reject split/schema drift before loading the expensive backbone."""
    if any(not rows.get(split) for split in ("train", "valid", "test")):
        raise ValueError("Profile evaluation requires non-empty train/valid/test splits")
    vocabulary = [str(value) for value in rows["train"][0]["kc_vocabulary"]]
    if len(vocabulary) < 4 or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("KC vocabulary must contain at least four unique entries")
    seen_ids: set[str] = set()
    for split in ("train", "valid", "test"):
        for row in rows[split]:
            if [str(value) for value in row["kc_vocabulary"]] != vocabulary:
                raise ValueError("Profile rows do not share one frozen KC vocabulary")
            if str(row.get("split")) != split:
                raise ValueError("Profile row split label does not match its file")
            student_id = str(row["student_id"])
            if student_id in seen_ids:
                raise ValueError("Student IDs overlap across profile splits")
            seen_ids.add(student_id)
            if int(row["state_history_length"]) != len(row["state_history"]):
                raise ValueError("Recorded state-history length is inconsistent")
            if not row["state_history"]:
                raise ValueError("State profile cannot be built from empty history")
            width_fields = (
                "prefix_counts",
                "prefix_correct",
                "future_counts",
                "future_correct",
                "future_eligible_mask",
            )
            if any(len(row[field]) != len(vocabulary) for field in width_fields):
                raise ValueError("Profile target width differs from KC vocabulary")
    return vocabulary


def expanded_feature_views(
    state: np.ndarray, projection_dim: int, projection_seed: int
) -> dict[str, np.ndarray]:
    """Retain complementary summaries without persisting the raw trajectory."""
    if state.ndim != 4:
        raise ValueError("State trajectory must be [N,L,S,D]")
    n, layers, slots, hidden = state.shape
    if layers % 8:
        raise ValueError("Depth-band views require a layer count divisible by eight")
    views = probe.feature_views(state, projection_dim, projection_seed)
    projection = probe.random_projection_matrix(hidden, projection_dim, projection_seed + 1)
    slot_projection_dim = max(4, projection_dim // 4)
    slot_projection = probe.random_projection_matrix(
        hidden, slot_projection_dim, projection_seed + 2
    )

    final_slots = np.einsum(
        "nsd,dp->nsp", state[:, -1], projection, optimize=True
    )
    views["final_layer_slots_rp"] = final_slots.reshape(n, -1)
    mean_depth_slots = np.einsum(
        "nsd,dp->nsp", state.mean(axis=1), projection, optimize=True
    )
    views["mean_layers_slots_rp"] = mean_depth_slots.reshape(n, -1)
    bands = state.reshape(n, 8, layers // 8, slots, hidden).mean(axis=2)
    band_slots = np.einsum(
        "nbsd,dp->nbsp", bands, projection, optimize=True
    )
    views["depth8_slots_rp"] = band_slots.reshape(n, -1)
    layer_slots = np.einsum(
        "nlsd,dp->nlsp", state, slot_projection, optimize=True
    )
    views["layerwise_slots_rp"] = layer_slots.reshape(n, -1)
    # np.linalg.vector_norm was added after the NumPy version in the approved
    # Feeds image. np.linalg.norm is identical here and works on NumPy 1.24+.
    views["layer_slot_norms"] = np.linalg.norm(state, axis=-1).reshape(n, -1)
    return {name: value.astype(np.float32) for name, value in views.items()}


@torch.inference_mode()
def extract_expanded_features(
    wrapper,
    tokenizer,
    rows: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    projection_dim: int,
    projection_seed: int,
) -> tuple[dict[str, np.ndarray], list[int]]:
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
        observed = list(state_numpy.shape[1:])
        if state_shape is None:
            state_shape = observed
        elif state_shape != observed:
            raise ValueError("Direct-state shape changed across batches")
        for name, values in expanded_feature_views(
            state_numpy, projection_dim, projection_seed
        ).items():
            outputs.setdefault(name, []).append(values)
        del state, state_numpy
        print(
            json.dumps({"states": min(start + batch_size, len(rows)), "total": len(rows)}),
            flush=True,
        )
    assert state_shape is not None
    return {
        name: np.concatenate(parts, axis=0) for name, parts in outputs.items()
    }, state_shape


def history_statistics(
    rows: list[dict[str, Any]], vocabulary: list[str], field: str
) -> np.ndarray:
    """Rates plus reliability from either visible history or full prefix."""
    rates = []
    counts_all = []
    for row in rows:
        if field == "state_history":
            index = {name: position for position, name in enumerate(vocabulary)}
            counts = np.zeros(len(vocabulary), dtype=np.float32)
            correct = np.zeros(len(vocabulary), dtype=np.float32)
            for event in row["state_history"]:
                for name in set(str(value) for value in event["kc_names"]):
                    if name not in index:
                        continue
                    position = index[name]
                    counts[position] += 1.0
                    correct[position] += float(event["correctness"])
        elif field == "prefix":
            counts = np.asarray(row["prefix_counts"], dtype=np.float32)
            correct = np.asarray(row["prefix_correct"], dtype=np.float32)
        else:
            raise ValueError(f"Unknown history field: {field}")
        rates.append((correct + 1.0) / (counts + 2.0))
        counts_all.append(np.log1p(counts))
    return np.concatenate((np.asarray(rates), np.asarray(counts_all)), axis=1)


def comparison_gates(
    metrics: dict[str, dict[str, dict[str, float]]],
    uncertainty: dict[str, Any],
    transport_summary: dict[str, Any],
) -> dict[str, bool]:
    """Centralize the preregistered directions and uncertainty thresholds."""
    state_metric = metrics["state8"]
    matched_metric = metrics["same_window_kc_outcomes"]
    hybrid_metric = metrics["state8_plus_same_window_kc_outcomes"]
    state_uncertainty = uncertainty["state8_vs_same_window_kc_outcomes"]
    rmse_delta = state_uncertainty["rmse_delta_model_minus_reference"]
    weak_delta = state_uncertainty["weak_f1_delta_model_minus_reference"]
    return {
        "state_beats_same_window_rmse": (
            state_metric["future_rate"]["rmse"]
            < matched_metric["future_rate"]["rmse"]
        ),
        "state_beats_same_window_weak_f1": (
            state_metric["weakness"]["weak_top2_macro_f1"]
            > matched_metric["weakness"]["weak_top2_macro_f1"]
        ),
        "hybrid_beats_same_window_rmse": (
            hybrid_metric["future_rate"]["rmse"]
            < matched_metric["future_rate"]["rmse"]
        ),
        "hybrid_beats_same_window_weak_f1": (
            hybrid_metric["weakness"]["weak_top2_macro_f1"]
            > matched_metric["weakness"]["weak_top2_macro_f1"]
        ),
        "state_rmse_bootstrap_probability_gt_0p95": (
            rmse_delta["bootstrap_probability_model_better"] > 0.95
        ),
        "state_weak_f1_bootstrap_probability_gt_0p95": (
            weak_delta["bootstrap_probability_model_better"] > 0.95
        ),
        "state_ownership_rmse_p_lt_0p05": (
            uncertainty["state_ownership_permutation_null"]["rmse"]["p_one_sided"]
            < 0.05
        ),
        "transport_support": transport_summary["clean"]["directions"] >= 30,
        "transport_specificity": (
            transport_summary["D_profile_specific"] is not None
            and transport_summary["D_profile_specific"] >= 0.10
        ),
    }


def normalize_views(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    normalized = ({}, {}, {})
    for name in train:
        train_value, mean, scale = probe.normalize_fit(train[name])
        normalized[0][name] = train_value
        normalized[1][name] = probe.normalize_apply(valid[name], mean, scale)
        normalized[2][name] = probe.normalize_apply(test[name], mean, scale)
    return normalized


def per_kc_validation_ensemble(
    train_views: dict[str, np.ndarray],
    valid_views: dict[str, np.ndarray],
    test_views: dict[str, np.ndarray],
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: list[float],
    min_valid_targets: int,
) -> dict[str, Any]:
    """Select one view and alpha per KC without consulting test outcomes."""
    train_views, valid_views, test_views = normalize_views(
        train_views, valid_views, test_views
    )
    train_y, train_mask, _ = targets["train"]
    valid_y, valid_mask, _ = targets["valid"]
    test_y, test_mask, _ = targets["test"]
    candidates: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    global_choice: tuple[float, str, float] | None = None
    for name in sorted(train_views):
        for alpha in alphas:
            weights, bias = probe.fit_weighted_ridge(
                train_views[name], train_y, train_mask, alpha
            )
            prediction = probe.predict_ridge(valid_views[name], weights, bias)
            candidates[(name, float(alpha))] = (weights, bias, prediction)
            rmse = probe.masked_metrics(prediction, valid_y, valid_mask)["rmse"]
            key = (float(rmse), name, float(alpha))
            if global_choice is None or key < global_choice:
                global_choice = key
    assert global_choice is not None
    fallback = (global_choice[1], global_choice[2])

    prediction = np.zeros_like(test_y, dtype=np.float64)
    validation_prediction = np.zeros_like(valid_y, dtype=np.float64)
    selections = []
    fit_y = np.concatenate((train_y, valid_y), axis=0)
    fit_mask = np.concatenate((train_mask, valid_mask), axis=0)
    for kc in range(train_y.shape[1]):
        train_n = int(train_mask[:, kc].sum())
        valid_n = int(valid_mask[:, kc].sum())
        choice = fallback
        selected_by = "global_validation_fallback"
        if train_n >= 2 and valid_n >= min_valid_targets:
            best: tuple[float, str, float] | None = None
            for (name, alpha), (_weights, _bias, values) in candidates.items():
                delta = values[valid_mask[:, kc], kc] - valid_y[valid_mask[:, kc], kc]
                key = (float(np.sqrt(np.square(delta).mean())), name, alpha)
                if best is None or key < best:
                    best = key
            assert best is not None
            choice = (best[1], best[2])
            selected_by = "per_kc_validation_rmse"
        name, alpha = choice
        validation_prediction[:, kc] = candidates[(name, alpha)][2][:, kc]
        fit_x = np.concatenate((train_views[name], valid_views[name]), axis=0)
        weights, bias = probe.fit_weighted_ridge(
            fit_x, fit_y[:, [kc]], fit_mask[:, [kc]], alpha
        )
        prediction[:, kc] = probe.predict_ridge(
            test_views[name], weights, bias
        )[:, 0]
        selections.append(
            {
                "kc_index": kc,
                "view": name,
                "alpha": float(alpha),
                "selected_by": selected_by,
                "train_targets": train_n,
                "validation_targets": valid_n,
                "test_targets": int(test_mask[:, kc].sum()),
            }
        )
    return {
        "prediction": prediction,
        "validation_prediction": validation_prediction,
        "future_rate": probe.masked_metrics(prediction, test_y, test_mask),
        "validation_future_rate": probe.masked_metrics(
            validation_prediction, valid_y, valid_mask
        ),
        "fallback_view": fallback[0],
        "fallback_alpha": float(fallback[1]),
        "selection": selections,
    }


def metric_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rows: list[dict[str, Any]],
    weakness_candidate_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if weakness_candidate_mask is None:
        weakness_mask = mask
    else:
        candidate = np.asarray(weakness_candidate_mask, dtype=bool)
        if candidate.ndim != 1 or candidate.shape[0] != target.shape[1]:
            raise ValueError("Weakness candidate mask must be one fixed [KC] vector")
        weakness_mask = np.broadcast_to(candidate, mask.shape)
    return {
        "rmse": float(probe.masked_metrics(prediction, target, mask)["rmse"]),
        "weak_top2_macro_f1": float(
            probe.weakness_metrics(prediction, rows, weakness_mask)["weak_top2_macro_f1"]
        ),
    }


def paired_student_bootstrap(
    prediction: np.ndarray,
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    weakness_candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = {"rmse_delta": [], "weak_f1_delta": []}
    for _ in range(replicates):
        indices = rng.integers(0, len(rows), size=len(rows))
        selected_rows = [rows[int(index)] for index in indices]
        model_metric = metric_pair(
            prediction[indices], target[indices], mask[indices], selected_rows,
            weakness_candidate_mask,
        )
        reference_metric = metric_pair(
            reference[indices], target[indices], mask[indices], selected_rows,
            weakness_candidate_mask,
        )
        draws["rmse_delta"].append(model_metric["rmse"] - reference_metric["rmse"])
        draws["weak_f1_delta"].append(
            model_metric["weak_top2_macro_f1"]
            - reference_metric["weak_top2_macro_f1"]
        )
    point_model = metric_pair(
        prediction, target, mask, rows, weakness_candidate_mask
    )
    point_reference = metric_pair(
        reference, target, mask, rows, weakness_candidate_mask
    )
    return {
        "replicates": replicates,
        "resampling_unit": "held-out student",
        "rmse_delta_model_minus_reference": {
            "value": point_model["rmse"] - point_reference["rmse"],
            "ci95": [float(x) for x in np.quantile(draws["rmse_delta"], [0.025, 0.975])],
            "bootstrap_probability_model_better": float(
                np.mean(np.asarray(draws["rmse_delta"]) < 0)
            ),
        },
        "weak_f1_delta_model_minus_reference": {
            "value": (
                point_model["weak_top2_macro_f1"]
                - point_reference["weak_top2_macro_f1"]
            ),
            "ci95": [float(x) for x in np.quantile(draws["weak_f1_delta"], [0.025, 0.975])],
            "bootstrap_probability_model_better": float(
                np.mean(np.asarray(draws["weak_f1_delta"]) > 0)
            ),
        },
    }


def permutation_null(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rows: list[dict[str, Any]],
    replicates: int,
    seed: int,
    weakness_candidate_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = metric_pair(
        prediction, target, mask, rows, weakness_candidate_mask
    )
    rmse = []
    weak = []
    for _ in range(replicates):
        permuted = prediction[rng.permutation(len(rows))]
        value = metric_pair(
            permuted, target, mask, rows, weakness_candidate_mask
        )
        rmse.append(value["rmse"])
        weak.append(value["weak_top2_macro_f1"])
    return {
        "replicates": replicates,
        "null": "permute held-out student-state ownership",
        "rmse": {
            "observed": observed["rmse"],
            "null_ci95": [float(x) for x in np.quantile(rmse, [0.025, 0.975])],
            "p_one_sided": float(
                (1 + np.sum(np.asarray(rmse) <= observed["rmse"])) / (replicates + 1)
            ),
        },
        "weak_top2_macro_f1": {
            "observed": observed["weak_top2_macro_f1"],
            "null_ci95": [float(x) for x in np.quantile(weak, [0.025, 0.975])],
            "p_one_sided": float(
                (1 + np.sum(np.asarray(weak) >= observed["weak_top2_macro_f1"]))
                / (replicates + 1)
            ),
        },
    }


def transport_pair_bootstrap(
    records: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    by_pair: dict[int, list[dict[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(int(row["pair_index"]), []).append(row)
    pair_ids = sorted(by_pair)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        sample = [row for pair_id in selected for row in by_pair[int(pair_id)]]
        value = v2.summarize_transport(sample)["D_profile_specific"]
        if value is not None:
            values.append(float(value))
    observed = v2.summarize_transport(records)["D_profile_specific"]
    return {
        "replicates": len(values),
        "resampling_unit": "KC-profile pair with both directions",
        "D_profile_specific": {
            "value": observed,
            "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])],
            "bootstrap_probability_positive": float(np.mean(np.asarray(values) > 0)),
        },
    }


def save_feature_cache(
    path: Path,
    features: dict[str, dict[str, np.ndarray]],
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        for name, value in features[split].items():
            arrays[f"{split}__{name}"] = value.astype(np.float16)
        rates, mask, counts = targets[split]
        arrays[f"{split}__future_rates"] = rates.astype(np.float32)
        arrays[f"{split}__future_mask"] = mask.astype(np.uint8)
        arrays[f"{split}__future_counts"] = counts.astype(np.float32)
    np.savez_compressed(path, **arrays)
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--recurrent-state", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-token-alias", default="<|vision_pad|>")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--state-batch-size", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--min-valid-targets", type=int, default=3)
    parser.add_argument("--min-transport-gap", type=float, default=0.30)
    parser.add_argument("--max-pairs-per-kc", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--permutation-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    args = parser.parse_args()

    rows = {
        split: probe.read_jsonl(args.manifest_dir / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }
    vocabulary = validate_rows(rows)
    tokenizer, wrapper = load_wrapper(
        args.model,
        args.adapter,
        args.recurrent_state,
        args.state_token_alias,
        "layerwise",
    )
    if wrapper.state_token_count != 8:
        raise ValueError("V3 requires the frozen eight-token state")

    features: dict[str, dict[str, np.ndarray]] = {}
    state_shape: list[int] | None = None
    for split in ("train", "valid", "test"):
        features[split], observed = extract_expanded_features(
            wrapper,
            tokenizer,
            rows[split],
            args.max_length,
            args.state_batch_size,
            args.projection_dim,
            args.seed,
        )
        if state_shape is None:
            state_shape = observed
        elif state_shape != observed:
            raise ValueError("State shape differs by split")
    targets = {split: probe.future_matrices(rows[split]) for split in rows}
    matched = {
        split: history_statistics(rows[split], vocabulary, "state_history")
        for split in rows
    }
    oracle = {
        split: history_statistics(rows[split], vocabulary, "prefix") for split in rows
    }

    state_fit = per_kc_validation_ensemble(
        features["train"],
        features["valid"],
        features["test"],
        targets,
        args.alphas,
        args.min_valid_targets,
    )
    matched_fit = per_kc_validation_ensemble(
        {"same_window_kc_outcomes": matched["train"]},
        {"same_window_kc_outcomes": matched["valid"]},
        {"same_window_kc_outcomes": matched["test"]},
        targets,
        args.alphas,
        args.min_valid_targets,
    )
    oracle_fit = per_kc_validation_ensemble(
        {"full_prefix_oracle": oracle["train"]},
        {"full_prefix_oracle": oracle["valid"]},
        {"full_prefix_oracle": oracle["test"]},
        targets,
        args.alphas,
        args.min_valid_targets,
    )
    hybrid_views = {
        split: {
            name: np.concatenate((value, matched[split]), axis=1)
            for name, value in features[split].items()
        }
        for split in features
    }
    hybrid_fit = per_kc_validation_ensemble(
        hybrid_views["train"],
        hybrid_views["valid"],
        hybrid_views["test"],
        targets,
        args.alphas,
        args.min_valid_targets,
    )

    test_y, test_mask, _ = targets["test"]
    train_y, train_mask, _ = targets["train"]
    population = probe.mean_predictor(train_y, train_mask, len(rows["test"]))
    state_prediction = state_fit["prediction"]
    matched_prediction = matched_fit["prediction"]
    oracle_prediction = oracle_fit["prediction"]
    hybrid_prediction = hybrid_fit["prediction"]
    train_supported_kcs = train_mask.sum(axis=0) >= 2

    transport, transport_summary = v2.transport_records(
        state_prediction,
        test_y,
        test_mask,
        vocabulary,
        seed=args.seed,
        min_gap=args.min_transport_gap,
        max_pairs_per_kc=args.max_pairs_per_kc,
    )
    profile_rows = []
    for index, (row, values, eligible) in enumerate(
        zip(rows["test"], state_prediction, test_mask, strict=True), start=1
    ):
        profile_rows.append(
            {
                "profile_id": f"heldout_profile_{index:04d}",
                "predicted": v2.readable_profile(
                    vocabulary, values, train_supported_kcs
                ),
                "evaluation_only_future_supported_prediction": v2.readable_profile(
                    vocabulary, values, eligible
                ),
                "future_validated_gold": {
                    "weak_indices": row["weak_kc_indices"],
                    "weaknesses": row["weak_kcs"],
                    "strengths": row["strong_kcs"],
                },
                "raw_history_visible_to_readout": False,
                "student_id_visible_to_readout": False,
            }
        )

    models = {
        "state8": state_prediction,
        "same_window_kc_outcomes": matched_prediction,
        "state8_plus_same_window_kc_outcomes": hybrid_prediction,
        "full_prefix_oracle": oracle_prediction,
        "train_population_mean": population,
    }
    metrics = {
        name: {
            "future_rate": probe.masked_metrics(value, test_y, test_mask),
            "weakness": probe.weakness_metrics(value, rows["test"], test_mask),
        }
        for name, value in models.items()
    }
    uncertainty = {
        "state8_vs_same_window_kc_outcomes": paired_student_bootstrap(
            state_prediction,
            matched_prediction,
            test_y,
            test_mask,
            rows["test"],
            args.bootstrap_replicates,
            args.seed + 1,
        ),
        "hybrid_vs_same_window_kc_outcomes": paired_student_bootstrap(
            hybrid_prediction,
            matched_prediction,
            test_y,
            test_mask,
            rows["test"],
            args.bootstrap_replicates,
            args.seed + 2,
        ),
        "state8_vs_full_prefix_oracle": paired_student_bootstrap(
            state_prediction,
            oracle_prediction,
            test_y,
            test_mask,
            rows["test"],
            args.bootstrap_replicates,
            args.seed + 3,
        ),
        "state_ownership_permutation_null": permutation_null(
            state_prediction,
            test_y,
            test_mask,
            rows["test"],
            args.permutation_replicates,
            args.seed + 4,
        ),
        "transport_pair_bootstrap": transport_pair_bootstrap(
            transport, args.bootstrap_replicates, args.seed + 5
        ),
    }
    gates = comparison_gates(metrics, uncertainty, transport_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = save_feature_cache(
        args.output_dir / "compressed_feature_cache.npz", features, targets
    )
    result = {
        "stage": STAGE,
        "status": "complete",
        "source": "DBE-KT22 real longitudinal interactions",
        "students": {split: len(rows[split]) for split in rows},
        "student_disjoint": True,
        "chronological_future_targets": True,
        "state_input_scope": "last 12 pre-cutoff interactions",
        "same_window_baseline_scope": (
            "KC names, correctness, and counts from the same last 12 "
            "pre-cutoff interactions"
        ),
        "same_window_baseline_limitation": (
            "interaction-window matched but not representation-capacity matched; "
            "State8 also receives question and response text"
        ),
        "oracle_scope": "entire pre-cutoff prefix; strictly more information than state",
        "raw_history_visible_to_state_readout": False,
        "student_id_visible_to_readout": False,
        "readout_protocol": (
            "frozen State8 -> validation-selected ridge heads -> structured readable "
            "profile template; no free-form language generation"
        ),
        "transport_protocol": (
            "fixed KC coordinate readout; only frozen state ownership changes; "
            "this is readout-level rather than end-to-end decoder transport"
        ),
        "state_token_alias": args.state_token_alias,
        "state_shape": state_shape,
        "state_token_count": wrapper.state_token_count,
        "metrics": metrics,
        "transport": transport_summary,
        "uncertainty": uncertainty,
        "selection": {
            "state8": state_fit["selection"],
            "same_window_kc_outcomes": matched_fit["selection"],
            "state8_plus_same_window_kc_outcomes": hybrid_fit["selection"],
            "full_prefix_oracle": oracle_fit["selection"],
            "minimum_validation_targets_per_kc": args.min_valid_targets,
            "test_labels_used_for_selection": False,
        },
        "feature_cache": {
            "raw_state_persisted": False,
            "arrays": cache_manifest,
        },
        "gates": gates,
        "scientific_gate_pass": all(gates.values()),
        "artifact_sha256": {
            "adapter_model": probe.sha256(Path(args.adapter) / "adapter_model.safetensors"),
            "recurrent_state": probe.sha256(args.recurrent_state),
        },
    }
    v2.write_jsonl(args.output_dir / "readable_profiles.jsonl", profile_rows)
    v2.write_jsonl(args.output_dir / "fixed_query_transport.jsonl", transport)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
