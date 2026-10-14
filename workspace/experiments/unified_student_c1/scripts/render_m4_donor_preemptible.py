#!/usr/bin/env python3
"""Render a preemption-tolerant donor-directional diagnostic.

The scientific objective, data order, seed, topology, and 500-round horizon are
unchanged.  Only the atomic checkpoint interval changes from 250 to 50 rounds.
"""

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
    return text.replace(old, new, 1)


def render(base: dict) -> dict:
    value = copy.deepcopy(base)
    jobs = value.get("jobs") or []
    if {job.get("name") for job in jobs} != {
        "m4-causal-donor-directional-sum-w1-round0to500-seed42",
        "m4-causal-donor-directional-sum-w4-round0to500-seed42",
    }:
        raise ValueError("Donor-directional source jobs drifted")
    value["description"] = (
        "Preemption-tolerant persistent donor-directional causal margin; "
        "scientific protocol unchanged, atomic checkpoints every 50 rounds"
    )
    for job in jobs:
        command = str(job["command"])
        command = replace_exact(command, "--save-steps 250", "--save-steps 50")
        command = replace_exact(
            command,
            "int(receipt['round']) % 250 == 0",
            "int(receipt['round']) % 50 == 0",
        )
        job["command"] = command
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output}")


if __name__ == "__main__":
    main()
