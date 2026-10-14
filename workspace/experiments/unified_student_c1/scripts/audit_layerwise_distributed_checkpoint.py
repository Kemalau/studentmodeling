#!/usr/bin/env python3
"""Fail-closed audit of an exact-epoch distributed recovery checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_ROWS = 259_845
EXPECTED_WORLD_SIZE = 16
EXPECTED_TOTAL_ROUNDS = math.ceil(EXPECTED_ROWS / EXPECTED_WORLD_SIZE)
EXPECTED_CAUSAL_PAIRS = 1_051
EXPECTED_CAUSAL_EXPOSURES = 12_992
EXPECTED_TRAIN_SHA256 = "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd"
EXPECTED_VALID_SHA256 = "78bfce75576dc3873cbb6459669a952adeedb8b77bda15f281af9519e2506d14"
EXPECTED_CAUSAL_TRAIN_SHA256 = "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766"
EXPECTED_CAUSAL_VALID_SHA256 = "8e038c3a60c2e1b420779cd54eb203b12ae5d51544c87a805fe19d47fdbe294e"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _audit_planned_optimizer_steps(config: dict[str, Any], *, expected_round: int) -> dict[str, Any]:
    """Accept full-run plans or a strictly evidenced promoted continuation plan.

    Promoted exact-epoch jobs resume from the selected formation checkpoint and
    therefore record the number of *remaining* optimizer steps in
    ``planned_optimizer_steps``.  The global scheduler/round/history continue to
    use full-epoch coordinates.  Do not accept an arbitrary shortened plan: the
    two independently recorded promotion source rounds, global stop round, and
    resume path must all agree.
    """

    actual = config.get("planned_optimizer_steps")
    if actual == EXPECTED_TOTAL_ROUNDS:
        return {
            "planned_optimizer_steps": int(actual),
            "planned_optimizer_steps_semantics": "full_epoch_total",
            "continuation_source_round": None,
        }

    source_round = config.get("promoted_real_order_source_round")
    objective_source_round = config.get("promoted_objective_transition_source_round")
    continuation_evidence = {
        "promoted_real_order_source_round": source_round,
        "promoted_objective_transition_source_round": objective_source_round,
        "distributed_rounds": config.get("distributed_rounds"),
        "resume_from_checkpoint": config.get("resume_from_checkpoint"),
        "diagnostic_weight_only_warm_start": config.get("diagnostic_weight_only_warm_start"),
        "validation_only_from_distributed_checkpoint": config.get(
            "validation_only_from_distributed_checkpoint"
        ),
    }
    valid_source = (
        isinstance(source_round, int)
        and not isinstance(source_round, bool)
        and 0 < source_round <= expected_round
        and objective_source_round == source_round
    )
    valid_context = (
        config.get("distributed_rounds") == EXPECTED_TOTAL_ROUNDS
        and bool(config.get("resume_from_checkpoint"))
        and config.get("diagnostic_weight_only_warm_start") in (None, False)
        and config.get("validation_only_from_distributed_checkpoint") in (None, False)
    )
    valid_remaining_plan = (
        valid_source
        and valid_context
        and actual == EXPECTED_TOTAL_ROUNDS - source_round
    )
    _require(
        valid_remaining_plan,
        "frozen training config drift: "
        + json.dumps(
            {
                "planned_optimizer_steps": {
                    "expected_full_epoch": EXPECTED_TOTAL_ROUNDS,
                    "expected_promoted_remaining": (
                        EXPECTED_TOTAL_ROUNDS - source_round
                        if isinstance(source_round, int) and not isinstance(source_round, bool)
                        else None
                    ),
                    "actual": actual,
                    "continuation_evidence": continuation_evidence,
                }
            },
            sort_keys=True,
            default=str,
        ),
    )
    return {
        "planned_optimizer_steps": int(actual),
        "planned_optimizer_steps_semantics": "remaining_after_promoted_exact_resume",
        "continuation_source_round": int(source_round),
    }


def audit(path: Path, *, expected_round: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), "checkpoint payload is not an object")
    _require(payload.get("distributed") is True, "checkpoint is not distributed")
    _require(payload.get("world_size") == EXPECTED_WORLD_SIZE, "world-size drift")
    _require(payload.get("round") == expected_round, "checkpoint-round drift")

    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    _require(isinstance(optimizer, dict), "optimizer state is missing")
    _require(bool(optimizer.get("param_groups")), "optimizer parameter groups are empty")
    _require(bool(optimizer.get("state")), "optimizer moments are empty")
    _require(isinstance(scheduler, dict), "scheduler state is missing")
    _require(scheduler.get("last_epoch") == expected_round, "scheduler position drift")

    rank_rng = payload.get("rank_rng_states")
    _require(isinstance(rank_rng, list), "per-rank RNG states are missing")
    _require(len(rank_rng) == EXPECTED_WORLD_SIZE, "per-rank RNG support drift")
    for rank, row in enumerate(rank_rng):
        _require(isinstance(row, dict) and row.get("rank") == rank, f"rank-{rank} RNG mismatch")
        for key in ("python_random_state", "torch_random_state", "cuda_random_state"):
            _require(key in row, f"rank-{rank} lacks {key}")

    real_order = payload.get("real_order")
    _require(isinstance(real_order, list), "real-order state is missing")
    _require(len(real_order) == EXPECTED_ROWS, "real-order length drift")
    _require(set(real_order) == set(range(EXPECTED_ROWS)), "real-order is not a permutation")

    raw_schedule = payload.get("causal_schedule")
    _require(isinstance(raw_schedule, dict), "causal schedule is missing")
    schedule = {int(key): int(value) for key, value in raw_schedule.items()}
    _require(len(schedule) == EXPECTED_CAUSAL_EXPOSURES, "causal-exposure count drift")
    _require(all(0 <= key < EXPECTED_ROWS for key in schedule), "causal position out of range")
    _require(
        all(0 <= value < EXPECTED_CAUSAL_PAIRS for value in schedule.values()),
        "causal pair index out of range",
    )
    _require(len(set(schedule.values())) == EXPECTED_CAUSAL_PAIRS, "causal pair coverage drift")

    history = payload.get("history")
    expected_history = min(EXPECTED_ROWS, expected_round * EXPECTED_WORLD_SIZE)
    _require(isinstance(history, list), "training history is missing")
    _require(len(history) == expected_history, "training-history support drift")
    positions = [int(row["global_position"]) for row in history]
    _require(positions == list(range(expected_history)), "training-history order drift")
    observed_causal = 0
    for row in history:
        position = int(row["global_position"])
        pair_index = schedule.get(position)
        pair_id = row.get("causal_pair_id")
        if pair_index is None:
            _require(pair_id is None, f"unexpected causal pair at position {position}")
            _require(float(row.get("causal_loss", 0.0)) == 0.0, f"unexpected causal loss at position {position}")
            continue
        observed_causal += 1
        _require(
            pair_id == f"recurrent-causal:{pair_index:07d}",
            f"causal pair mismatch at position {position}",
        )
        causal_loss = float(row.get("causal_loss", float("nan")))
        _require(math.isfinite(causal_loss), f"non-finite causal loss at position {position}")
        _require(
            float(row.get("causal_effective_weight", 0.0)) > 0.0,
            f"inactive causal weight at position {position}",
        )
        _require(
            row.get("causal_objective_phase") == "specificity_triplet",
            f"causal objective drift at position {position}",
        )
    expected_observed_causal = sum(position < expected_history for position in schedule)
    _require(
        observed_causal == expected_observed_causal,
        "observed causal-history support drift",
    )

    config = payload.get("config")
    _require(isinstance(config, dict), "frozen training config is missing")
    optimizer_plan = _audit_planned_optimizer_steps(config, expected_round=expected_round)
    expected = {
        "state_architecture": "layerwise",
        "layerwise_operation_checkpointing": True,
        "layerwise_transition_rank": 256,
        "distributed": True,
        "distributed_world_size": EXPECTED_WORLD_SIZE,
        "real_train_rows": EXPECTED_ROWS,
        "causal_train_pairs": EXPECTED_CAUSAL_PAIRS,
        "planned_causal_exposures": EXPECTED_CAUSAL_EXPOSURES,
        "causal_objective_mode": "specificity_triplet",
        "causal_repeat_fraction": 0.05,
        "normalize_causal_exposure": True,
        "state_token_alias": "<|vision_pad|>",
        "causal_branch_dropout_mode": "eval_deterministic",
        "train_sha256": EXPECTED_TRAIN_SHA256,
        "valid_sha256": EXPECTED_VALID_SHA256,
        "causal_train_sha256": EXPECTED_CAUSAL_TRAIN_SHA256,
        "causal_valid_sha256": EXPECTED_CAUSAL_VALID_SHA256,
        "seed": 42,
    }
    drift = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    _require(not drift, "frozen training config drift: " + json.dumps(drift, sort_keys=True))

    return {
        "stage": "layerwise_distributed_checkpoint_recovery_audit",
        "status": "passed",
        "exact_resume_ready": True,
        "checkpoint_round": expected_round,
        "world_size": EXPECTED_WORLD_SIZE,
        "optimizer_state_present": True,
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "rank_rng_states": len(rank_rng),
        "real_order_rows": len(real_order),
        "history_rows": len(history),
        "observed_causal_history_rows": observed_causal,
        "causal_exposures": len(schedule),
        "unique_causal_pairs": len(set(schedule.values())),
        "full_epoch_rounds": EXPECTED_TOTAL_ROUNDS,
        **optimizer_plan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.checkpoint, expected_round=args.expected_round)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
