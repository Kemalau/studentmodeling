#!/usr/bin/env python3
"""Render a checkpoint-specific layerwise C4 probe from the generated template."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "layerwise-finalfit-checkpoint1000-c4-seed42"
SOURCE_RUN_DIR = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "layerwise_unified_finalfit_seed42_20260801_snapshot"
)


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}")
    return text.replace(old, new)


def render(
    base: dict,
    checkpoint_round: int,
    *,
    run_dir: str = SOURCE_RUN_DIR,
    job_name: str | None = None,
) -> dict:
    if checkpoint_round <= 0:
        raise ValueError("Checkpoint round must be positive")
    value = copy.deepcopy(base)
    value["description"] = (
        "Diagnostic-only frozen C4 probe of the formal layerwise unified fit "
        f"at atomic round {checkpoint_round}"
    )
    jobs = value.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != SOURCE_JOB:
        raise ValueError("Source layerwise checkpoint C4 template drift")
    job = jobs[0]
    job["name"] = job_name or f"layerwise-finalfit-checkpoint{checkpoint_round}-c4-seed42"
    command = str(job["command"])
    command = replace_once(command, f"export RUN_DIR={SOURCE_RUN_DIR}", f"export RUN_DIR={run_dir}")
    command = replace_once(command, "checkpoint-round-1000", f"checkpoint-round-{checkpoint_round}")
    command = replace_once(
        command,
        "receipt.get('round') == 1000",
        f"receipt.get('round') == {checkpoint_round}",
    )
    command = replace_once(command, "--expected-round 1000", f"--expected-round {checkpoint_round}")
    command = replace_once(command, "checkpoint_round=1000", f"checkpoint_round={checkpoint_round}")
    job["command"] = command
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", default=SOURCE_RUN_DIR)
    parser.add_argument("--job-name")
    args = parser.parse_args()
    value = render(
        load(args.base),
        args.round,
        run_dir=args.run_dir,
        job_name=args.job_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output} round={args.round}")


if __name__ == "__main__":
    main()
