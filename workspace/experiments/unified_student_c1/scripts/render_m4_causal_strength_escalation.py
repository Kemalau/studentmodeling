#!/usr/bin/env python3
"""Render the preregistered second-stage M4 causal-strength search."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


TRAIN_SHA256 = "f03c363cd3614965b495cab73629c8ab79eebce4d8dc72a873e205cc1212f4bc"
PREVIOUS_TRAIN_SHA256 = "74b33cb1cc3da4e76f1fb77f89090d8e7a1b94cd0be125ab36afcdad16e2b08b"
LEGACY_TRAIN_SHA256 = "24f066362c951edb932caf2b98c4cf909b7a211cc090a7a3623946d659bfc1d2"


SPECS = {
    "raw8": {
        "job": "m4-causal-raw8-round250-to500-seed42",
        "weight": "8.0",
        "normalize": "0",
    },
    "mean2_normalized": {
        "job": "m4-causal-mean2-normalized-round250-to500-seed42",
        "weight": "2.0",
        "normalize": "1",
    },
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return value


def render(base: dict, branches: list[str]) -> dict:
    jobs = base.get("jobs") or []
    source = next(
        (job for job in jobs if job.get("name") == "m4-causal-raw4-round250-to500-seed42"),
        None,
    )
    if source is None:
        raise ValueError("Raw4 source job is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "Second-stage controlled M4 causal-strength search from atomic round 250 "
        "to 500 after the lower-dose branches miss the frozen C4 gate"
    )
    result["jobs"] = []
    for branch in branches:
        spec = SPECS[branch]
        job = copy.deepcopy(source)
        job["name"] = spec["job"]
        env = job["submit_args"]["env"]
        env["BRANCH"] = branch
        env["CAUSAL_WEIGHT"] = spec["weight"]
        env["NORMALIZE_CAUSAL"] = spec["normalize"]
        command = job.get("command")
        command_is_list = isinstance(command, list)
        if command_is_list:
            if len(command) != 1 or not isinstance(command[0], str):
                raise ValueError(f"Unsupported command shape in {spec['job']}")
            command = command[0]
        if not isinstance(command, str):
            raise ValueError(f"Missing command in {spec['job']}")
        old_hashes = [
            value
            for value in (LEGACY_TRAIN_SHA256, PREVIOUS_TRAIN_SHA256)
            if command.count(value) == 1
        ]
        if len(old_hashes) != 1 or command.count(TRAIN_SHA256):
            raise ValueError("Expected exactly one previous training-script hash")
        command = command.replace(old_hashes[0], TRAIN_SHA256, 1)
        old_branches = "assert branch in {'raw4', 'mean_normalized'}, branch"
        new_branches = "assert branch in {'raw8', 'mean2_normalized'}, branch"
        if command.count(old_branches) != 1:
            raise ValueError("Expected exactly one legacy checkpoint-branch assertion")
        command = command.replace(old_branches, new_branches, 1)
        job["command"] = [command] if command_is_list else command
        result["jobs"].append(job)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branches", nargs="*", choices=tuple(SPECS))
    args = parser.parse_args()
    branches = args.branches or list(SPECS)
    value = render(load(args.base), branches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(json.dumps({"branches": branches, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
