#!/usr/bin/env python3
"""Render a checkpoint-specific C4 probe from the generated round-250 template."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def replace_exact(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}")
    return text.replace(old, new)


def render(
    base: dict,
    checkpoint_round: int,
    *,
    run_dir: str | None = None,
    job_name: str | None = None,
) -> dict:
    if checkpoint_round <= 0:
        raise ValueError("Checkpoint round must be positive")
    value = copy.deepcopy(base)
    value["description"] = (
        f"Frozen 150-pair C4 diagnostic of atomic M4 round-{checkpoint_round} checkpoint"
    )
    jobs = value.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != "m4-checkpoint250-c4-seed42":
        raise ValueError("Source checkpoint probe template drift")
    job = jobs[0]
    job["name"] = job_name or f"m4-checkpoint{checkpoint_round}-c4-seed42"
    command = str(job["command"])
    command = replace_exact(
        command, "checkpoint-round-250", f"checkpoint-round-{checkpoint_round}"
    )
    command = replace_exact(
        command,
        "receipt.get('round') == 250",
        f"receipt.get('round') == {checkpoint_round}",
    )
    command = replace_exact(
        command, "checkpoint_round=250", f"checkpoint_round={checkpoint_round}"
    )
    if run_dir is not None:
        command = replace_exact(
            command,
            "export RUN_DIR=/mnt/amulet/studentmodeling/checkpoints/world_model_v1_m4_generation_kd_trainable_seed42_fixedu4_v2_20260730",
            f"export RUN_DIR={run_dir}",
        )
    job["command"] = command
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--job-name")
    args = parser.parse_args()
    value = render(
        load(args.base), args.round, run_dir=args.run_dir, job_name=args.job_name
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output} round={args.round}")


if __name__ == "__main__":
    main()
