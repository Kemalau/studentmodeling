#!/usr/bin/env python3
"""Stage a T1+T2 Trainer checkpoint locally and tighten save cadence only."""

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


def stage(source: Path, destination: Path, *, save_steps: int, max_steps: int) -> dict:
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete source checkpoint: {missing}")
    source_state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(source_state.get("global_step", -1))
    if not 0 < step < max_steps or int(source_state.get("max_steps", -1)) != max_steps:
        raise ValueError(f"Incompatible Trainer horizon: step={step}, state={source_state}")
    if save_steps <= 0:
        raise ValueError("save_steps must be positive")

    before = {name: digest(source / name) for name in REQUIRED}
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    copied = {name: digest(destination / name) for name in REQUIRED}
    if copied != before:
        raise ValueError("Checkpoint staging hash mismatch")

    state_path = destination / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    old_save_steps = int(state.get("save_steps", -1))
    state["save_steps"] = save_steps
    control = (state.get("stateful_callbacks") or {}).get("TrainerControl")
    if control is not None:
        control.setdefault("args", {})["should_save"] = False
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    after = {name: digest(destination / name) for name in REQUIRED}
    for name in REQUIRED:
        if name != "trainer_state.json" and after[name] != before[name]:
            raise ValueError(f"Cadence migration changed scientific payload {name}")
    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    if int(migrated["global_step"]) != step or int(migrated["save_steps"]) != save_steps:
        raise ValueError("Cadence migration verification failed")
    return {
        "stage": "t1_t2_paired_addone_checkpoint_cadence_migration_v1",
        "status": "valid",
        "source": str(source),
        "destination": str(destination),
        "global_step": step,
        "max_steps": max_steps,
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
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=40692)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    value = stage(
        args.source,
        args.destination,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.destination)


if __name__ == "__main__":
    main()
