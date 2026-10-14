#!/usr/bin/env python3
"""Fail-closed validation for the 50-round donor checkpoint variant."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from amlt_validation_utils import command_text
from validate_m4_causal_bootstrap_rehearsal_amlt_config import (
    load as load_base,
    validate as validate_base,
)


def validate(config: dict, selected: set[str] | None = None) -> None:
    jobs = config.get("jobs") or []
    if not jobs:
        raise ValueError("Missing donor jobs")
    normalized = copy.deepcopy(config)
    for original, copied in zip(jobs, normalized["jobs"], strict=True):
        command = command_text(original.get("command"))
        if command.count("--save-steps 50") != 1 or "--save-steps 250" in command:
            raise ValueError("Preemption checkpoint interval drift")
        marker50 = "int(receipt['round']) % 50 == 0"
        marker250 = "int(receipt['round']) % 250 == 0"
        if command.count(marker50) != 1 or marker250 in command:
            raise ValueError("Checkpoint receipt alignment drift")
        copied_command = command.replace("--save-steps 50", "--save-steps 250", 1)
        copied_command = copied_command.replace(marker50, marker250, 1)
        copied["command"] = copied_command
    validate_base(normalized, selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(
        load_base(args.config),
        set(args.jobs) if args.jobs else None,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
