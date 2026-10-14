#!/usr/bin/env python3
"""Recover fail-closed diagnostic provenance from a hash-receipted checkpoint."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit(
    checkpoint: Path,
    receipt_path: Path,
    adapter_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    receipt = read(receipt_path)
    manifest = read(manifest_path)
    if receipt.get("status") != "complete" or receipt.get("round") != 3250:
        raise ValueError("Checkpoint completion receipt drift")
    if receipt.get("world_size") != 4:
        raise ValueError("Checkpoint world-size drift")
    observed = {
        "adapter_model": sha256(adapter_path),
        "recurrent_state": sha256(checkpoint),
    }
    if observed != receipt.get("sha256"):
        raise ValueError("Checkpoint payload hash drift")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = payload.get("config")
    history = payload.get("history")
    real_order = payload.get("real_order")
    causal_schedule = payload.get("causal_schedule")
    rank_rng_states = payload.get("rank_rng_states")
    optimizer = payload.get("optimizer")
    scheduler = payload.get("scheduler")
    if not isinstance(config, dict) or not isinstance(history, list):
        raise ValueError("Checkpoint config/history missing")
    if not isinstance(real_order, list) or len(real_order) != 259845:
        raise ValueError("Checkpoint full real order missing")
    if not isinstance(causal_schedule, dict) or len(causal_schedule) != 12992:
        raise ValueError("Checkpoint causal schedule drift")
    if not isinstance(rank_rng_states, list) or len(rank_rng_states) != 4:
        raise ValueError("Checkpoint rank RNG support drift")
    if not isinstance(optimizer, dict) or not isinstance(scheduler, dict):
        raise ValueError("Checkpoint optimizer/scheduler missing")

    scripts = Path(__file__).resolve().parent
    trainer_sha256 = sha256(scripts / "train_recurrent_student_m4.py")
    entrypoint_sha256 = sha256(scripts / "train_recurrent_student_m4_open_t3.py")
    expected_trainer_sha256 = (
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440"
    )
    expected_entrypoint_sha256 = (
        "b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a"
    )
    if trainer_sha256 != expected_trainer_sha256:
        raise ValueError("Frozen diagnostic trainer SHA256 drift")
    if entrypoint_sha256 != expected_entrypoint_sha256:
        raise ValueError("Open-T3 diagnostic entrypoint SHA256 drift")

    positions = [int(row["global_position"]) for row in history]
    example_ids = [str(row["example_id"]) for row in history]
    tasks = collections.Counter(str(row["task"]) for row in history)
    sources = collections.Counter(str(row["source"]) for row in history)
    required_config = {
        "state_architecture": "layerwise",
        "state_token_alias": "<|vision_pad|>",
        "learning_rate": 5e-7,
        "behavior_kd_weight": 0.0,
        "direct_replay_weight": 0.0,
        "causal_weight": 2.0,
        "causal_objective_mode": "specificity_triplet",
        "causal_repeat_fraction": 0.05,
        "layerwise_transition_rank": 256,
        "diagnostic_start_global_position": 12000,
        "diagnostic_stop_after_round": 3250,
        "diagnostic_index_manifest_sha256": sha256(manifest_path),
        "diagnostic_weight_only_warm_start": True,
        "diagnostic_optimizer_reset": True,
        "diagnostic_warm_start_source_world_size": 16,
        "diagnostic_warm_start_source_round": 750,
        "distributed_world_size": 4,
        "seed": 42,
    }
    failed = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required_config.items()
        if config.get(key) != expected
    }
    if failed:
        raise ValueError(f"Checkpoint scientific recipe drift: {failed}")
    expected_mirror = (
        "/mnt/amulet/studentmodeling/diagnostics/"
        "layerwise_task_retention_service_recovery/equal_t3_t6_open_t3/state_path/"
        f"lr5e-7_start12000_stop3250_{sha256(manifest_path)}_"
        f"{expected_trainer_sha256}_{expected_entrypoint_sha256}"
    )
    if str(config.get("checkpoint_mirror_dir")) != expected_mirror:
        raise ValueError("Checkpoint mirror provenance drift")
    if positions != list(range(12000, 13000)):
        raise ValueError("Checkpoint diagnostic position support drift")
    if len(example_ids) != 1000 or len(set(example_ids)) != 1000:
        raise ValueError("Checkpoint diagnostic example support drift")
    if tasks != collections.Counter(manifest.get("task_counts") or {}):
        raise ValueError("Checkpoint task counts disagree with manifest")
    if sources != collections.Counter(manifest.get("source_counts") or {}):
        raise ValueError("Checkpoint source counts disagree with manifest")

    return {
        "stage": "layerwise_diagnostic_checkpoint_provenance_audit",
        "status": "passed",
        "branch": "equal_t3_t6_open_t3",
        "loss_profile": "state_path",
        "learning_rate": 5e-7,
        "causal_weight": 2.0,
        "checkpoint_round": 3250,
        "current_world_size": 4,
        "source_checkpoint_round": 750,
        "source_world_size": 16,
        "history_rows": 1000,
        "history_global_position_start": 12000,
        "history_global_position_end": 12999,
        "history_unique_examples": 1000,
        "causal_pairs_seen": 50,
        "causal_schedule_size": 12992,
        "full_real_order_rows": 259845,
        "rank_rng_states": 4,
        "optimizer_present": True,
        "scheduler_present": True,
        "manifest_sha256": sha256(manifest_path),
        "task_counts": dict(tasks),
        "source_counts": dict(sources),
        "formal_example_id_overlap": manifest.get("formal_example_id_overlap"),
        "training_view": "open_t3_no_candidate_enumeration",
        "trainer_sha256": trainer_sha256,
        "train_entrypoint": "scripts/train_recurrent_student_m4_open_t3.py",
        "train_entrypoint_sha256": entrypoint_sha256,
        "checkpoint_mirror_dir": expected_mirror,
        "sha256": observed,
        "config": {key: config[key] for key in required_config},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.receipt, args.adapter, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
