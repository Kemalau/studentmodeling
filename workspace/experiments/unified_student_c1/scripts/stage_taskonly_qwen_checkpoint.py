#!/usr/bin/env python3
"""Stage an immutable task-only checkpoint and migrate save cadence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REQUIRED = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stage(source: Path, destination: Path, *, expected_step: int, save_steps: int) -> dict:
    if destination.exists():
        shutil.rmtree(destination)
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete source checkpoint: {missing}")
    before = {name: digest(source / name) for name in REQUIRED}
    shutil.copytree(source, destination)
    copied = {name: digest(destination / name) for name in REQUIRED}
    if copied != before:
        raise ValueError("Checkpoint staging hash mismatch")

    state_path = destination / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != expected_step:
        raise ValueError(f"global_step drift: {state.get('global_step')} != {expected_step}")
    old_save_steps = int(state.get("save_steps", -1))
    if old_save_steps not in (50, 250):
        raise ValueError(f"Unexpected source save_steps={old_save_steps}")
    state["save_steps"] = save_steps
    control = (state.get("stateful_callbacks") or {}).get("TrainerControl")
    if control is not None:
        args = control.get("args") or {}
        args["should_save"] = False
        control["args"] = args
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    after = {name: digest(destination / name) for name in REQUIRED}
    unchanged = [name for name in REQUIRED if name != "trainer_state.json"]
    if any(after[name] != before[name] for name in unchanged):
        raise ValueError("Cadence migration changed a scientific checkpoint payload")
    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    if int(migrated["global_step"]) != expected_step or int(migrated["save_steps"]) != save_steps:
        raise ValueError("Cadence migration receipt verification failed")
    return {
        "stage": "taskonly_qwen_checkpoint_cadence_migration_v1",
        "status": "valid",
        "source": str(source),
        "destination": str(destination),
        "global_step": expected_step,
        "old_save_steps": old_save_steps,
        "new_save_steps": save_steps,
        "source_hashes": before,
        "staged_hashes": after,
        "only_trainer_state_changed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    value = stage(
        args.source,
        args.destination,
        expected_step=args.expected_step,
        save_steps=args.save_steps,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.destination)


if __name__ == "__main__":
    main()
