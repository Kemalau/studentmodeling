#!/usr/bin/env python3
"""Fail-closed validation for the CF4 recurrent causal diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_cf4_diagnostic import (
    SPECS,
    TRAIN_JSONL_SHA256,
    TRAIN_MANIFEST,
    TRAIN_MANIFEST_SHA256,
)


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    jobs = config.get("jobs") or []
    all_expected = {spec["job"] for spec in SPECS.values()}
    expected = expected_jobs if expected_jobs is not None else all_expected
    if not expected or not expected.issubset(all_expected):
        raise ValueError(f"Unknown or empty expected job set: {sorted(expected)}")
    if {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("Job set drift")
    by_job = {spec["job"]: (branch, spec) for branch, spec in SPECS.items()}
    for job in jobs:
        name = str(job["name"])
        branch, spec = by_job[name]
        if (
            job.get("sku") != "80G4-A100"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("BRANCH") != branch
            or str(env.get("CAUSAL_WEIGHT")) != "1.0"
        ):
            raise ValueError(f"Retry/UAI/branch drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            TRAIN_MANIFEST,
            TRAIN_MANIFEST_SHA256,
            TRAIN_JSONL_SHA256,
            "recurrent_causal_pairs_cf4_balanced_v1",
            f"--causal-objective-mode {spec['objective']}",
            "--causal-preference-reduction sum",
            "--causal-repeat-fraction 1.0",
            "--diagnostic-stop-after-round 250",
            "--save-steps 50",
            "--distributed-world-size 4",
            "micro_steps') == 1000",
            "causal_pairs_seen') == 1000",
            "unique_causal_pairs_seen') == 1000",
            "export CUDA_VISIBLE_DEVICES=0",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
            "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Forbidden path/secret in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
