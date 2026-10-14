#!/usr/bin/env python3
"""Bind the approved 19-job final evaluator to the matched canonical model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


CANONICAL = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_diagnosis_first_matched_unified_canonical_v1_20260810"
)
PROGRESS = "state8_diagnosis_first_matched_final_full259845"


def materialize(source: Path, output: Path) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    jobs = config.get("jobs") or []
    if len(jobs) != 19:
        raise ValueError(f"Expected 19 final-evaluation jobs, observed {len(jobs)}")
    for job in jobs:
        env = (job.setdefault("submit_args", {})).setdefault("env", {})
        env["STATE8_FINAL_EVAL_RUN_DIR"] = CANONICAL
        env["STATE8_FINAL_EVAL_PROGRESS_NAMESPACE"] = PROGRESS
    config["description"] = (
        "Diagnosis-first matched State8 exact-epoch final evaluation: "
        "16 formal shards and three answer/diagnosis C4 seeds"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
