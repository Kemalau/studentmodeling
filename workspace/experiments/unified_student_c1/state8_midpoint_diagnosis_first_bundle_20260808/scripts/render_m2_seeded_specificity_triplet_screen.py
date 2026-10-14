#!/usr/bin/env python3
"""Render the C4-aligned random-negative screen from the matched M2 CF4 job."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


SOURCE_JOB = "m2-seeded-cf4-donor-directional-single-gb4-step0to1000-seed42"
JOB = "m2-seeded-cf4-specificity-triplet-v2-single-gb4-step0to1000-seed42"
SOURCE_BRANCH = "m2_seeded_donor_directional_cf4_w1_single_gb4"
BRANCH = "m2_seeded_cf4_specificity_triplet_v2_w1_single_gb4"
SOURCE_RUN_DIR = "/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m2_seeded_cf4_donor_directional_w1_seed42_step1000_20260731"
RUN_DIR = "/mnt/amulet/studentmodeling/diagnostics/world_model_v1_m2_seeded_cf4_specificity_triplet_v2_w1_seed42_step1000_20260731"
TRAINER_SHA256 = "ab25b40f2b4e91582c9a3bd8bd784161f44aaa6ff5c7aa4c6d19da81693bac32"
EVALUATOR_SHA256 = "d2a0cf7702f3aaf56bc3629af7b63837f739799f47a3b51e5f5076917116f0b2"
LEGACY_TRAINER_SHA256 = "e9d1046730aef7bd2afaaae82c730a79123b386eaf95715ae3880e169414894e"
LEGACY_EVALUATOR_SHA256 = "9dca3271d29153baf66b440239467e6889c0f2c64677c4e31fd4e2d8a5710c10"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"Expected one source fragment, found {count}: {old!r}")
    return text.replace(old, new)


def render(base: dict) -> dict:
    source = next(
        (job for job in base.get("jobs") or [] if job.get("name") == SOURCE_JOB),
        None,
    )
    if source is None:
        raise ValueError("Source M2-seeded CF4 job is missing")
    result = copy.deepcopy(base)
    result["description"] = (
        "M2 CF1-seeded C4-specificity screen with the frozen norm-matched "
        "random state as an explicit negative; strict state-only decoder"
    )
    job = copy.deepcopy(source)
    job["name"] = JOB
    command = str(job["command"])
    command = command.replace(LEGACY_TRAINER_SHA256, TRAINER_SHA256)
    command = command.replace(LEGACY_EVALUATOR_SHA256, EVALUATOR_SHA256)
    command = replace_once(command, SOURCE_RUN_DIR, RUN_DIR)
    command = command.replace(SOURCE_BRANCH, BRANCH)
    if SOURCE_BRANCH in command:
        raise ValueError("Source branch remained after rendering")
    command = command.replace(
        "--causal-objective-mode donor_directional",
        "--causal-objective-mode specificity_triplet",
    )
    command = command.replace(
        "config.get('causal_objective_mode') == 'donor_directional'",
        "config.get('causal_objective_mode') == 'specificity_triplet'",
    )
    required = (
        RUN_DIR,
        BRANCH,
        "--causal-objective-mode specificity_triplet",
        "config.get('causal_objective_mode') == 'specificity_triplet'",
        TRAINER_SHA256,
        EVALUATOR_SHA256,
        "initialization=m2_cf1_causal_adapter_and_state",
    )
    if any(value not in command for value in required):
        raise ValueError("Specificity-triplet command transformation is incomplete")
    forbidden = (
        SOURCE_RUN_DIR,
        SOURCE_BRANCH,
        "--causal-objective-mode donor_directional",
    )
    if any(value in command for value in forbidden):
        raise ValueError("Source causal objective leaked into rendered command")
    job["command"] = command
    env = job.setdefault("submit_args", {}).setdefault("env", {})
    env["BRANCH"] = BRANCH
    result["jobs"] = [job]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = render(load(args.base))
    args.output.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(f"rendered={args.output}")


if __name__ == "__main__":
    main()
