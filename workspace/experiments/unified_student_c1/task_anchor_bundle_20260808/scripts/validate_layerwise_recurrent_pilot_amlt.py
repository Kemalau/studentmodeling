#!/usr/bin/env python3
"""Fail-closed AMLT validator for the layerwise recurrent pilot."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "layerwise-pilot-preserve-low-lr-seed42": {
        "BRANCH": "preserve_low_lr",
        "LEARNING_RATE": "5e-7",
        "CAUSAL_WEIGHT": "0.5",
        "CAUSAL_OBJECTIVE": "donor_directional",
    },
    "layerwise-pilot-balanced-seed42": {
        "BRANCH": "balanced",
        "LEARNING_RATE": "2e-6",
        "CAUSAL_WEIGHT": "1.0",
        "CAUSAL_OBJECTIVE": "donor_directional",
    },
    "layerwise-pilot-specificity-seed42": {
        "BRANCH": "specificity",
        "LEARNING_RATE": "2e-6",
        "CAUSAL_WEIGHT": "1.0",
        "CAUSAL_OBJECTIVE": "specificity_triplet",
    },
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
    if {job.get("name") for job in jobs} != set(JOBS) or len(jobs) != 3:
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
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        for key, expected in JOBS[name].items():
            if str(env.get(key)) != expected:
                raise ValueError(f"Branch env drift: {name}/{key}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "--init-recurrent-state",
            "--trainable-projections",
            "sha256sum scripts/recurrent_student_model.py",
        )
        if any(value.lower() in command.lower() for value in forbidden):
            raise ValueError("Secret/local/scalar path leaked into pilot")
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Pilot branches do not share one controlled command")
    command = commands[0]
    required = (
        "35579bf50fb5ed84de70d7f88d683b0cf2e939b66cef9a6b629abb27b2bbe2a4",
        "1706a01450660c833534e487a8b7aad56a61541ce41cc17ca2c8ed16e2e23d3c",
        "d2a0cf7702f3aaf56bc3629af7b63837f739799f47a3b51e5f5076917116f0b2",
        "--state-architecture layerwise",
        "--layerwise-operation-checkpointing",
        "--distributed-mpi --distributed-world-size 4",
        "OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required",
        "LOCAL_RANK=0",
        "OMPI_COMM_WORLD_SIZE:?MPI world size is required",
        "--diagnostic-stop-after-round 50",
        "--causal-repeat-fraction 1.0",
        "--causal-objective-mode",
        "--state-architecture layerwise \\",
        "cf4_balanced_paired_state_swap_train_seed20260731_v1.json",
        "cf4_balanced_paired_state_swap_valid_seed20260801_v1.json",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/world_model_v1/m4/teacher_cache_train_n12000_top256_fixedu4_v2_20260730",
        "--behavior-kd-weight 0.5",
        "flock -x 7",
        "flock -x 6",
        "/tmp/layerwise-pilot-data.complete",
        "/tmp/layerwise-pilot-model.complete",
        "/mnt/amulet/studentmodeling/checkpoints/layerwise_recurrent_pilot_",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "causal_branch_dropout_mode'] == 'eval_deterministic",
        "layerwise_operation_checkpointing'] is True",
    )
    if any(value not in command for value in required):
        raise ValueError("Layerwise pilot command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
