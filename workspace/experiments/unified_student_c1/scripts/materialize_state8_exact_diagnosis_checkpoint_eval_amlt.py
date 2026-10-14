#!/usr/bin/env python3
"""Derive a hash-bound diagnosis-first checkpoint screen from the step-2000 config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


ALLOWED_STEPS = {2000, 3000, 4000, 5000}


def materialize(source: Path, output: Path, step: int) -> None:
    if step not in ALLOWED_STEPS or step == 2000:
        raise ValueError("Derived checkpoint screen step must be one of 3000/4000/5000")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    jobs = config.get("jobs") or []
    if len(jobs) != 2:
        raise ValueError("Expected the two-job step-2000 source screen")
    config["description"] = (
        f"Exact State8 diagnosis-first checkpoint-{step} semantic T3 and causal C4 screen"
    )
    for job in jobs:
        name = str(job.get("name") or "")
        if "step2000" not in name:
            raise ValueError(f"Unexpected source job name: {name}")
        job["name"] = name.replace("step2000", f"step{step}")
        env = (job.get("submit_args") or {}).get("env") or {}
        root = str(env.get("STATE8_EXACT_CHECKPOINT_ROOT") or "")
        if not root.endswith("/checkpoint-2000"):
            raise ValueError(f"Unexpected source checkpoint root: {root}")
        env["STATE8_EXACT_CHECKPOINT_ROOT"] = root[: -len("checkpoint-2000")] + f"checkpoint-{step}"
        env["STATE8_EXACT_CHECKPOINT_STEP"] = str(step)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(partial, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=sorted(ALLOWED_STEPS - {2000}), required=True)
    args = parser.parse_args()
    materialize(args.source, args.output, args.step)
    print(f"output={args.output} step={args.step}")


if __name__ == "__main__":
    main()
