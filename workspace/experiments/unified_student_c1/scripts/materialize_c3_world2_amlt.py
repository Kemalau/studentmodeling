#!/usr/bin/env python3
"""Mechanically derive the schedulable C3 world2/global-batch4 config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from validate_c3_exposure_matched_amlt import load, validate
except ImportError:
    from scripts.validate_c3_exposure_matched_amlt import load, validate


def materialize(source: Path, output: Path) -> dict:
    config = load(source)
    config["description"] = (
        "Exposure-matched C3 transfer continuations; world2 global-batch4 "
        "schedulability recovery"
    )
    for job in config.get("jobs") or []:
        job["sku"] = "80G2-A100"
        env = job["submit_args"]["env"]
        env["C3_WORLD_SIZE"] = "2"
        env["C3_GRADIENT_ACCUMULATION_STEPS"] = "2"
        env["DIAGNOSTIC_STOP_AFTER_ROUND"] = "12500"
    validate(config, expected_topology="world2")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
