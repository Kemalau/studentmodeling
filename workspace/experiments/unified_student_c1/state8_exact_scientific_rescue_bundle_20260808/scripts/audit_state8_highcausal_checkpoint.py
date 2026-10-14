#!/usr/bin/env python3
"""Audit a 4-GPU State8 high-causal screening checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


ROWS = 1000
WORLD_SIZE = 4
ROUNDS = 250
CAUSAL_PAIRS = 250
CAUSAL_EXPOSURES = 1000
TRAIN_SHA256 = "2a484f43515c111a62cef1328299acd00eb21e35efc81cd8f48b94fb36a162f0"
VALID_SHA256 = "78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14"
CAUSAL_TRAIN_SHA256 = "5d4cef18e5bbbc253d9be23b06283dc44cad7a00684062dd8f5d26cca3e2868a"
CAUSAL_VALID_SHA256 = "55e4b5212e45f9ac6a3b369cb665817e2c1a9cbf903ce0866ac8aad18e4ede50"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(path: Path, *, expected_round: int, variant: str) -> dict[str, Any]:
    if variant not in {"joint", "updater", "mid", "eedi", "taxonomy"}:
        raise ValueError("Unknown high-causal variant")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "checkpoint payload is not an object")
    require(payload.get("distributed") is True, "checkpoint is not distributed")
    require(payload.get("world_size") == WORLD_SIZE, "world-size drift")
    require(payload.get("round") == expected_round, "checkpoint-round drift")
    require(0 < expected_round <= ROUNDS, "expected round is outside the screen")

    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    require(isinstance(optimizer, dict) and bool(optimizer.get("state")), "optimizer moments are missing")
    require(bool(optimizer.get("param_groups")), "optimizer parameter groups are missing")
    require(isinstance(scheduler, dict), "scheduler state is missing")
    require(scheduler.get("last_epoch") == expected_round, "scheduler position drift")

    rng = payload.get("rank_rng_states")
    require(isinstance(rng, list) and len(rng) == WORLD_SIZE, "rank RNG support drift")
    require([row.get("rank") for row in rng] == list(range(WORLD_SIZE)), "rank RNG order drift")
    for rank, row in enumerate(rng):
        for key in ("python_random_state", "torch_random_state", "cuda_random_state"):
            require(key in row, f"rank-{rank} lacks {key}")

    order = payload.get("real_order")
    require(isinstance(order, list) and len(order) == ROWS, "real-order length drift")
    require(set(order) == set(range(ROWS)), "real-order is not a permutation")

    raw_schedule = payload.get("causal_schedule")
    require(isinstance(raw_schedule, dict), "causal schedule is missing")
    schedule = {int(key): int(value) for key, value in raw_schedule.items()}
    require(len(schedule) == CAUSAL_EXPOSURES, "causal exposure count drift")
    require(set(schedule) == set(range(ROWS)), "causal schedule must cover every training row")
    require(all(0 <= value < CAUSAL_PAIRS for value in schedule.values()), "causal pair index drift")
    require(len(set(schedule.values())) == CAUSAL_PAIRS, "causal pair coverage drift")

    history = payload.get("history")
    expected_history = expected_round * WORLD_SIZE
    require(isinstance(history, list) and len(history) == expected_history, "history support drift")
    require(
        [int(row["global_position"]) for row in history] == list(range(expected_history)),
        "history order drift",
    )
    phase = {
        "joint": "semantic_random_neutral_specificity",
        "mid": "semantic_random_neutral_specificity",
        "updater": "updater_only_native_donor",
        "eedi": "semantic_random_neutral_specificity",
        "taxonomy": "semantic_random_neutral_specificity",
    }[variant]
    causal_ids: set[str] = set()
    geometry_rows = 0
    for row in history:
        position = int(row["global_position"])
        pair = schedule[position]
        pair_id = str(row.get("causal_pair_id"))
        require(
            pair_id == f"state8-qualified-causal:{pair:07d}:diagnosis",
            f"causal pair mismatch at position {position}",
        )
        causal_ids.add(pair_id)
        for key in ("loss", "generation_loss", "online_kd_loss", "causal_loss"):
            require(math.isfinite(float(row.get(key, float("nan")))), f"non-finite {key} at {position}")
        require(row.get("causal_objective_phase") == phase, f"causal phase drift at {position}")
        require(float(row.get("history_dropout", -1)) == 1.0, f"history dropout drift at {position}")
        require(row.get("history_dropped") is True, f"history visible at {position}")
        require(row.get("recurrent_state_present") is True, f"state missing at {position}")
        require(float(row.get("causal_effective_weight", 0.0)) > 0.0, f"causal path inactive at {position}")
        geometry_rows += int("semantic_geometry_effective_weight" in row)

    config = payload.get("config")
    require(isinstance(config, dict), "frozen config is missing")
    expected = {
        "state_architecture": "layerwise",
        "state_token_count": 8,
        "state_token_alias": "<|vision_pad|>",
        "layerwise_operation_checkpointing": True,
        "layerwise_transition_rank": 256,
        "distributed": True,
        "distributed_world_size": WORLD_SIZE,
        "planned_optimizer_steps": ROUNDS,
        "real_train_rows": ROWS,
        "train_sha256": (
            "176a5a5fabd2c4a5d9d660ade7ae15db86aecf119c8f78f388a9f4c3b29b8be7"
            if variant in {"eedi", "taxonomy"}
            else TRAIN_SHA256
        ),
        "valid_sha256": VALID_SHA256,
        "causal_train_pairs": CAUSAL_PAIRS,
        "planned_causal_exposures": CAUSAL_EXPOSURES,
        "causal_train_sha256": CAUSAL_TRAIN_SHA256,
        "causal_valid_sha256": CAUSAL_VALID_SHA256,
        "causal_objective_mode": "specificity_triplet",
        "causal_preference_reduction": "sum",
        "causal_repeat_fraction": 1.0,
        "normalize_causal_exposure": True,
        "causal_margin": 1.0,
        "history_dropout_schedule": [1.0],
        "final_inference_route": "strict_state_only",
        "output_schema": "label_diagnosis_evidence_v1",
        "online_kd_weight": 1.0,
        "label_auxiliary_weight": 4.0,
        "semantic_causal_label_weight": 8.0,
        "semantic_different_max_cosine": 0.75,
        "semantic_auxiliary_visible_at_inference": False,
        "external_state_teacher_trainable": False,
        "teacher_trainable": False,
        "consistency_weight": 0.1,
    }
    variant_expected = {
        "joint": {
            "seed": 47,
            "causal_weight": 8.0,
            "semantic_geometry_weight": 1.0,
            "semantic_random_neutral": True,
            "semantic_random_target": "insufficient_evidence",
            "semantic_updater_only": False,
            "layerwise_transition_only": False,
        },
        "updater": {
            "seed": 48,
            "causal_weight": 8.0,
            "semantic_geometry_weight": 1.0,
            "semantic_random_neutral": True,
            "semantic_random_target": "held_out_evaluation_only",
            "semantic_updater_only": True,
            "layerwise_transition_only": True,
        },
        "mid": {
            "seed": 49,
            "causal_weight": 4.0,
            "semantic_geometry_weight": 0.5,
            "semantic_random_neutral": True,
            "semantic_random_target": "insufficient_evidence",
            "semantic_updater_only": False,
            "layerwise_transition_only": False,
        },
        "eedi": {
            "seed": 50,
            "causal_weight": 4.0,
            "semantic_geometry_weight": 0.5,
            "eedi_semantic_geometry_weight": 0.25,
            "semantic_random_neutral": True,
            "semantic_random_target": "insufficient_evidence",
            "semantic_updater_only": False,
            "layerwise_transition_only": False,
        },
        "taxonomy": {
            "seed": 51,
            "causal_weight": 4.0,
            "semantic_geometry_weight": 0.5,
            "eedi_semantic_geometry_weight": 0.25,
            "semantic_random_neutral": True,
            "semantic_random_target": "insufficient_evidence",
            "semantic_updater_only": False,
            "layerwise_transition_only": False,
            "taxonomy_label_contract": "source_aware_no_answer_labels_v1",
            "taxonomy_candidates_visible_to_decoder": False,
        },
    }[variant]
    expected.update(variant_expected)
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    require(not drift, "frozen high-causal config drift: " + json.dumps(drift, sort_keys=True))
    require(geometry_rows > 0, "semantic geometry branch was never observed")
    return {
        "stage": "state8_highcausal_checkpoint_audit_v1",
        "status": "passed",
        "exact_resume_ready": True,
        "variant": variant,
        "checkpoint_round": expected_round,
        "world_size": WORLD_SIZE,
        "optimizer_state_present": True,
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "rank_rng_states": len(rng),
        "real_order_rows": len(order),
        "history_rows": len(history),
        "causal_exposures": len(schedule),
        "unique_causal_pairs": len(set(schedule.values())),
        "observed_unique_causal_pairs": len(causal_ids),
        "semantic_geometry_history_rows": geometry_rows,
        "final_inference_route": config["final_inference_route"],
        "state_token_count": config["state_token_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-round", type=int, required=True)
    parser.add_argument(
        "--variant", choices=("joint", "updater", "mid", "eedi", "taxonomy"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.checkpoint, expected_round=args.expected_round, variant=args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
