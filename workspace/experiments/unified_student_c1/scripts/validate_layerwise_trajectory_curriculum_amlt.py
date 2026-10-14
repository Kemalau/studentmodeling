#!/usr/bin/env python3
"""Fail-closed validator for the frozen-CF1 trajectory curriculum."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "trajectory-r64-scale-seed42": ("r64_scale", "64", "1.0"),
    "trajectory-r256-scale-seed42": ("r256_scale", "256", "1.0"),
    "trajectory-r256-cosine-seed42": ("r256_cosine", "256", "0.0"),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 3 or {job.get("name") for job in jobs} != set(JOBS):
        raise ValueError("Job set drift")
    commands = []
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G4-A100"
            or job.get("sla_tier") != "Standard"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        expected = JOBS[name]
        observed = (
            str(env.get("BRANCH")),
            str(env.get("TRANSITION_RANK")),
            str(env.get("RELATIVE_MSE_WEIGHT")),
        )
        if observed != expected:
            raise ValueError(f"Branch env drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Curriculum branches do not share one controlled command")
    command = commands[0]
    required = (
        "a39d0d1aabe97e7df3bc0424e18a556af338ba6bf4f328ea21b921915bd4727f",
        "--state-architecture layerwise",
        "--layerwise-transition-only",
        "--trajectory-curriculum-only",
        "--trajectory-relative-mse-weight",
        "--generation-weight 0 --behavior-kd-weight 0",
        "--direct-replay-weight 0 --causal-weight 0 --causal-trajectory-weight 1.0",
        "--distributed-mpi --distributed-world-size 4",
        "--diagnostic-stop-after-round 50",
        "assert gradient['lora']['parameter_tensors'] == 0",
        "/mnt/amulet/studentmodeling/checkpoints/layerwise_trajectory_curriculum_",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
    )
    if any(value not in command for value in required):
        raise ValueError("Trajectory curriculum command drift")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--trainable-projections",
    )
    if any(value.lower() in command.lower() for value in forbidden):
        raise ValueError("Secret/local/scalar path leaked into curriculum")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
