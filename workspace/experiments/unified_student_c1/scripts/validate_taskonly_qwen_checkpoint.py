#!/usr/bin/env python3
"""Fail-closed validation and selection of resumable task-only checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def validate(path: Path, *, expected_max_steps: int | None = None) -> dict:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid checkpoint directory name: {path}")
    step = int(match.group(1))
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: {missing}")
    trainer = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    adapter = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if int(trainer.get("global_step", -1)) != step:
        raise ValueError(f"Checkpoint/global-step drift: {path}")
    max_steps = int(trainer.get("max_steps", -1))
    if expected_max_steps is not None and max_steps != expected_max_steps:
        raise ValueError(f"Max-step drift: {max_steps} != {expected_max_steps}")
    if not 0 < step < max_steps:
        raise ValueError(f"Checkpoint step is not resumable: {step}/{max_steps}")
    target_modules = set(adapter.get("target_modules") or [])
    expected_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if int(adapter.get("r", -1)) != 8 or int(adapter.get("lora_alpha", -1)) != 16:
        raise ValueError("LoRA rank/alpha drift")
    if target_modules != expected_modules or adapter.get("peft_type") != "LORA":
        raise ValueError("LoRA target-module/type drift")
    if (path / "adapter_model.safetensors").stat().st_size < 250_000_000:
        raise ValueError("Adapter artifact is unexpectedly small")
    if (path / "optimizer.pt").stat().st_size < 500_000_000:
        raise ValueError("Optimizer artifact is unexpectedly small")
    return {
        "stage": "taskonly_qwen_checkpoint_validation_v1",
        "status": "valid",
        "path": str(path.resolve()),
        "global_step": step,
        "max_steps": max_steps,
        "remaining_steps": max_steps - step,
        "files": {name: (path / name).stat().st_size for name in REQUIRED_FILES},
    }


def select(root: Path, *, expected_max_steps: int | None = None) -> dict:
    valid = []
    errors = {}
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            valid.append(validate(path, expected_max_steps=expected_max_steps))
        except Exception as error:  # report all rejected candidates
            errors[path.name] = str(error)
    if not valid:
        raise ValueError(f"No valid checkpoint under {root}; rejected={errors}")
    selected = max(valid, key=lambda value: value["global_step"])
    selected["rejected"] = errors
    selected["candidate_count"] = len(valid)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-max-steps", type=int)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    value = select(args.run_dir, expected_max_steps=args.expected_max_steps)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(value["path"])


if __name__ == "__main__":
    main()
