#!/usr/bin/env python3
"""Decode and transport future-validated profiles from a frozen State8.

The encoder sees only the pre-cutoff interaction prefix.  Ridge readouts are
fit on disjoint training students and selected on disjoint validation students.
All reported profile targets come from post-cutoff behavior of held-out test
students.  The transport panel freezes the queried KC and changes only the
student state supplied to the readout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import evaluate_dbe_student_profile_probe as probe
    from evaluate_recurrent_formal_protocols import load_wrapper
except ModuleNotFoundError:  # Package-style imports in local unit tests.
    from scripts import evaluate_dbe_student_profile_probe as probe
    from scripts.evaluate_recurrent_formal_protocols import load_wrapper


STAGE = "frozen_state8_future_profile_transport_v2"


def fit_view(
    train_x: np.ndarray,
    valid_x: np.ndarray,
    test_x: np.ndarray,
    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    alphas: list[float],
) -> dict[str, Any]:
    train_x, mean, scale = probe.normalize_fit(train_x)
    valid_x = probe.normalize_apply(valid_x, mean, scale)
    test_x = probe.normalize_apply(test_x, mean, scale)
    train_y, train_mask, _ = targets["train"]
    valid_y, valid_mask, _ = targets["valid"]
    test_y, test_mask, _ = targets["test"]
    alpha, train_weights, train_bias = probe.choose_alpha(
        train_x,
        train_y,
        train_mask,
        valid_x,
        valid_y,
        valid_mask,
        alphas,
    )
    valid_prediction = probe.predict_ridge(valid_x, train_weights, train_bias)
    fit_x = np.concatenate((train_x, valid_x), axis=0)
    fit_y = np.concatenate((train_y, valid_y), axis=0)
    fit_mask = np.concatenate((train_mask, valid_mask), axis=0)
    weights, bias = probe.fit_weighted_ridge(fit_x, fit_y, fit_mask, alpha)
    prediction = probe.predict_ridge(test_x, weights, bias)
    return {
        "selected_alpha": float(alpha),
        "feature_dim": int(test_x.shape[1]),
        "validation_future_rate": probe.masked_metrics(
            valid_prediction, valid_y, valid_mask
        ),
        "future_rate": probe.masked_metrics(prediction, test_y, test_mask),
        "prediction": prediction,
        "test_x": test_x,
        "weights": weights,
        "bias": bias,
    }


def readable_profile(
    vocabulary: list[str], prediction: np.ndarray, eligible: np.ndarray
) -> dict[str, Any]:
    supported = np.flatnonzero(eligible)
    if len(supported) < 4:
        raise ValueError("Readable profile requires at least four supported KCs")
    ordered = supported[np.argsort(prediction[supported], kind="stable")]
    weak = [int(value) for value in ordered[:2]]
    strong = [int(value) for value in ordered[-2:][::-1]]

    def field(indices: list[int]) -> list[dict[str, Any]]:
        return [
            {
                "kc": vocabulary[index],
                "predicted_future_success": round(float(prediction[index]), 6),
            }
            for index in indices
        ]

    weak_fields = field(weak)
    strong_fields = field(strong)
    text = (
        "<student_profile>\n"
        f"<weakness>{'; '.join(value['kc'] for value in weak_fields)}</weakness>\n"
        f"<strength>{'; '.join(value['kc'] for value in strong_fields)}</strength>\n"
        "<basis>Decoded only from the frozen State8; the readout cannot access "
        "student ID or raw interaction history.</basis>\n"
        "</student_profile>"
    )
    return {
        "weak_indices": weak,
        "strong_indices": strong,
        "weaknesses": weak_fields,
        "strengths": strong_fields,
        "text": text,
    }


def build_transport_pairs(
    target: np.ndarray,
    mask: np.ndarray,
    *,
    min_gap: float,
    max_pairs_per_kc: int,
) -> list[tuple[int, int, int]]:
    """Return (KC, low-student, high-student) pairs from held-out outcomes."""
    pairs: list[tuple[int, int, int]] = []
    for kc in range(target.shape[1]):
        supported = np.flatnonzero(mask[:, kc])
        ordered = supported[np.argsort(target[supported, kc], kind="stable")]
        left = 0
        right = len(ordered) - 1
        local = 0
        while left < right and local < max_pairs_per_kc:
            low = int(ordered[left])
            high = int(ordered[right])
            if float(target[high, kc] - target[low, kc]) < min_gap:
                break
            pairs.append((kc, low, high))
            local += 1
            left += 1
            right -= 1
    if not pairs:
        raise ValueError("No future-profile transport pairs satisfy the gap")
    return pairs


def summarize_transport(records: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = ("noop", "same_profile", "random", "swap")
    panels: dict[str, Any] = {}
    for subset_name, selected in (
        ("all", records),
        ("clean", [row for row in records if row["native_closer_to_recipient"]]),
    ):
        panel: dict[str, Any] = {"directions": len(selected)}
        for condition in conditions:
            values = [row["conditions"][condition] for row in selected]
            panel[condition] = {
                "flip_to_donor_rate": (
                    float(np.mean([value["closer_to_donor"] for value in values]))
                    if values
                    else None
                ),
                "mean_directional_progress": (
                    float(np.mean([value["directional_progress"] for value in values]))
                    if values
                    else None
                ),
            }
        panels[subset_name] = panel
    clean = panels["clean"]
    swap = clean["swap"]["flip_to_donor_rate"]
    random = clean["random"]["flip_to_donor_rate"]
    same = clean["same_profile"]["flip_to_donor_rate"]
    panels["D_profile_specific"] = None if swap is None else float(swap - random)
    panels["D_profile_same_control"] = None if swap is None else float(swap - same)
    return panels


def transport_records(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    vocabulary: list[str],
    *,
    seed: int,
    min_gap: float,
    max_pairs_per_kc: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    pairs = build_transport_pairs(
        target,
        mask,
        min_gap=min_gap,
        max_pairs_per_kc=max_pairs_per_kc,
    )
    for pair_index, (kc, low, high) in enumerate(pairs, start=1):
        supported = [
            int(value)
            for value in np.flatnonzero(mask[:, kc])
            if int(value) not in (low, high)
        ]
        if len(supported) < 2:
            continue
        for direction, (recipient, donor) in enumerate(((low, high), (high, low))):
            native_target = float(target[recipient, kc])
            donor_target = float(target[donor, kc])
            gap = abs(donor_target - native_target)
            sign = float(np.sign(donor_target - native_target))
            same_index = min(
                supported,
                key=lambda index: (
                    abs(float(target[index, kc]) - native_target),
                    abs(float(target[index, kc]) - donor_target),
                    index,
                ),
            )
            random_index = int(rng.choice(supported))
            native_prediction = float(prediction[recipient, kc])
            condition_indices = {
                "noop": recipient,
                "same_profile": same_index,
                "random": random_index,
                "swap": donor,
            }
            conditions: dict[str, Any] = {}
            for condition, state_index in condition_indices.items():
                value = float(prediction[state_index, kc])
                conditions[condition] = {
                    "prediction": value,
                    "closer_to_donor": bool(
                        abs(value - donor_target) < abs(value - native_target)
                    ),
                    "directional_progress": float(
                        sign * (value - native_prediction) / gap
                    ),
                }
            records.append(
                {
                    "pair_index": pair_index,
                    "direction": direction,
                    "kc_index": kc,
                    "kc": vocabulary[kc],
                    "recipient_profile_index": recipient,
                    "donor_profile_index": donor,
                    "same_profile_control_index": same_index,
                    "random_control_index": random_index,
                    "recipient_future_success": native_target,
                    "donor_future_success": donor_target,
                    "future_success_gap": gap,
                    "native_prediction": native_prediction,
                    "native_closer_to_recipient": bool(
                        abs(native_prediction - native_target)
                        < abs(native_prediction - donor_target)
                    ),
                    "query_held_fixed": True,
                    "only_state_identity_changes": True,
                    "conditions": conditions,
                }
            )
    return records, summarize_transport(records)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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
    parser.add_argument("--min-transport-gap", type=float, default=0.30)
    parser.add_argument("--max-pairs-per-kc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260816)
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
        raise ValueError("This transport test requires the frozen eight-token state")

    features: dict[str, dict[str, np.ndarray]] = {}
    state_shape: list[int] | None = None
    for split in ("train", "valid", "test"):
        features[split], observed = probe.extract_direct_features(
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

    fitted = {
        name: fit_view(
            features["train"][name],
            features["valid"][name],
            features["test"][name],
            targets,
            args.alphas,
        )
        for name in features["train"]
    }
    selected_name = min(
        fitted,
        key=lambda name: (
            fitted[name]["validation_future_rate"]["rmse"],
            name,
        ),
    )
    selected = fitted[selected_name]
    test_y, test_mask, _ = targets["test"]
    prediction = selected["prediction"]
    train_supported_kcs = targets["train"][1].sum(axis=0) >= 2
    if int(train_supported_kcs.sum()) < 4:
        raise ValueError("Fewer than four KCs support a learned profile readout")

    prefix = {split: probe.prefix_rate_features(rows[split]) for split in rows}
    prefix_fit = fit_view(
        prefix["train"],
        prefix["valid"],
        prefix["test"],
        targets,
        args.alphas,
    )
    train_y, train_mask, _ = targets["train"]
    population = probe.mean_predictor(train_y, train_mask, len(rows["test"]))
    permutation = np.random.default_rng(args.seed + 17).permutation(len(rows["test"]))
    permuted_prediction = prediction[permutation]

    profile_rows = []
    for index, (row, values, eligible) in enumerate(
        zip(rows["test"], prediction, test_mask, strict=True), start=1
    ):
        # The deployable text ranks the complete frozen vocabulary.  The
        # post-cutoff test support mask is used only by the separate evaluation
        # metrics and is never exposed to the emitted profile. KCs lacking two
        # training targets are excluded because their constant fallback is not
        # a learned student-specific readout.
        decoded = readable_profile(vocabulary, values, train_supported_kcs)
        supported_decoded = readable_profile(vocabulary, values, eligible)
        profile_rows.append(
            {
                "profile_id": f"heldout_profile_{index:04d}",
                "selected_state_view": selected_name,
                "raw_history_visible_to_readout": False,
                "student_id_visible_to_readout": False,
                "predicted": decoded,
                "evaluation_only_future_supported_prediction": supported_decoded,
                "future_validated_gold": {
                    "weak_indices": row["weak_kc_indices"],
                    "weaknesses": row["weak_kcs"],
                    "strengths": row["strong_kcs"],
                },
            }
        )

    transport, transport_summary = transport_records(
        prediction,
        test_y,
        test_mask,
        vocabulary,
        seed=args.seed,
        min_gap=args.min_transport_gap,
        max_pairs_per_kc=args.max_pairs_per_kc,
    )
    state_weakness = probe.weakness_metrics(prediction, rows["test"], test_mask)
    prefix_weakness = probe.weakness_metrics(
        prefix_fit["prediction"], rows["test"], test_mask
    )
    population_weakness = probe.weakness_metrics(population, rows["test"], test_mask)
    permutation_weakness = probe.weakness_metrics(
        permuted_prediction, rows["test"], test_mask
    )
    selected_rmse = float(selected["future_rate"]["rmse"])
    prefix_rmse = float(prefix_fit["future_rate"]["rmse"])
    population_rmse = float(probe.masked_metrics(population, test_y, test_mask)["rmse"])
    permutation_rmse = float(
        probe.masked_metrics(permuted_prediction, test_y, test_mask)["rmse"]
    )
    clean_n = int(transport_summary["clean"]["directions"])
    d_specific = transport_summary["D_profile_specific"]
    gates = {
        "beats_prefix_rmse": selected_rmse < prefix_rmse,
        "beats_prefix_weak_f1": (
            state_weakness["weak_top2_macro_f1"]
            > prefix_weakness["weak_top2_macro_f1"]
        ),
        "beats_population_rmse": selected_rmse < population_rmse,
        "beats_population_weak_f1": (
            state_weakness["weak_top2_macro_f1"]
            > population_weakness["weak_top2_macro_f1"]
        ),
        "beats_permuted_state_rmse": selected_rmse < permutation_rmse,
        "beats_permuted_state_weak_f1": (
            state_weakness["weak_top2_macro_f1"]
            > permutation_weakness["weak_top2_macro_f1"]
        ),
        "transport_support": clean_n >= 30,
        "transport_specificity": d_specific is not None and d_specific >= 0.10,
    }
    result = {
        "stage": STAGE,
        "status": "complete",
        "source": "DBE-KT22 real longitudinal interactions",
        "students": {split: len(rows[split]) for split in rows},
        "student_disjoint": True,
        "chronological_future_targets": True,
        "raw_history_visible_to_readout": False,
        "student_id_visible_to_readout": False,
        "state_shape": state_shape,
        "state_token_count": wrapper.state_token_count,
        "train_supported_kcs": [
            vocabulary[index] for index in np.flatnonzero(train_supported_kcs)
        ],
        "constant_fallback_kcs": [
            vocabulary[index] for index in np.flatnonzero(~train_supported_kcs)
        ],
        "selected_state_view": selected_name,
        "selection_rule": "minimum validation future-rate RMSE; test untouched",
        "state": {
            "future_rate": selected["future_rate"],
            "weakness": state_weakness,
        },
        "baselines": {
            "prefix_kc_beta_rates": {
                "future_rate": prefix_fit["future_rate"],
                "weakness": prefix_weakness,
            },
            "train_population_mean": {
                "future_rate": probe.masked_metrics(population, test_y, test_mask),
                "weakness": population_weakness,
            },
            "permuted_test_states": {
                "future_rate": probe.masked_metrics(
                    permuted_prediction, test_y, test_mask
                ),
                "weakness": permutation_weakness,
            },
        },
        "compression_gap_vs_prefix_rmse": selected_rmse - prefix_rmse,
        "transport": transport_summary,
        "transport_protocol": {
            "query": "one fixed DBE knowledge component per pair",
            "recipient_and_donor_min_future_rate_gap": args.min_transport_gap,
            "only_state_identity_changes": True,
            "controls": ["noop", "same_profile", "random_state_owner"],
        },
        "gates": gates,
        "scientific_gate_pass": all(gates.values()),
        "artifact_sha256": {
            "adapter_model": probe.sha256(Path(args.adapter) / "adapter_model.safetensors"),
            "recurrent_state": probe.sha256(args.recurrent_state),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "readable_profiles.jsonl", profile_rows)
    write_jsonl(args.output_dir / "fixed_query_transport.jsonl", transport)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
