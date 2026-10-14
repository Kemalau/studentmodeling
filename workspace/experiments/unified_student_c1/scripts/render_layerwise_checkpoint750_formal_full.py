#!/usr/bin/env python3
"""Render full-support T2/T3/T5 evaluation of the immutable round-750 state."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from render_layerwise_checkpoint250_formal_screen import screen_command


SOURCE_JOBS = {
    "layerwise-final-eval-t2-seed42": "layerwise-checkpoint750-full-t2-seed42",
    "layerwise-final-eval-t3-seed42": "layerwise-checkpoint750-full-t3-seed42",
    "layerwise-final-eval-t5-seed42": "layerwise-checkpoint750-full-t5-seed42",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def full_command(command: str) -> str:
    value = screen_command(command)
    value = value.replace("checkpoint-round-1000", "checkpoint-round-750")
    value = value.replace("receipt.get('round')==1000", "receipt.get('round')==750")
    value = value.replace("--max-samples-per-group 25 \\\n        ", "")
    value = value.replace(
        "stage=layerwise_finalfit_checkpoint1000_formal_screen",
        "stage=layerwise_finalfit_checkpoint750_formal_full",
    )
    if "checkpoint-round-1000" in value or "--max-samples-per-group" in value:
        raise ValueError("Small-screen constraint survived full-support rendering")
    return value


def render(base: dict) -> dict:
    value = copy.deepcopy(base)
    value["description"] = (
        "Read-only full-support strict-history-free T2/T3/T5 evaluation of the "
        "immutable layerwise round-750 checkpoint"
    )
    by_name = {str(job.get("name")): job for job in value.get("jobs") or []}
    jobs = []
    for source_name, target_name in SOURCE_JOBS.items():
        source = by_name.get(source_name)
        if source is None:
            raise ValueError(f"Missing source job: {source_name}")
        job = copy.deepcopy(source)
        job["name"] = target_name
        job["command"] = full_command(str(job["command"]))
        submit = job.setdefault("submit_args", {})
        submit["max_attempts"] = 2
        submit["max_run_duration_seconds"] = 259200
        jobs.append(job)
    value["jobs"] = jobs
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output} jobs={len(value['jobs'])}")


if __name__ == "__main__":
    main()
