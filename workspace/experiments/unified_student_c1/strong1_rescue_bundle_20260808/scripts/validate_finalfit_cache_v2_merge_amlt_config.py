#!/usr/bin/env python3
"""Fail closed on the remote 16-way teacher-cache merge job."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ROOT = "/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
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
    if len(jobs) != 1 or jobs[0].get("name") != "finalfit-cache-v2-merge-seed42":
        raise ValueError("Job drift")
    job = jobs[0]
    if job.get("sku") != "80G1-A100" or job.get("mpi") is not False:
        raise ValueError("GPU/MPI drift")
    if int(job.get("process_count_per_node", 0)) != 1 or job.get("sla_tier") != "Standard":
        raise ValueError("Process/SLA drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    if submit.get("max_attempts") != 2 or int(submit.get("max_run_duration_seconds", 0)) != 14400:
        raise ValueError("Retry/duration drift")
    if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, str(job["name"]))
    required = (
        ROOT,
        "8bd00421608fe2c31df67ce624d2452183461877f9aa07a1e85d8552bd72c468",
        "merge_teacher_cache_partitions.py",
        "--expected-rows 259845",
        "partition_count'] == 16",
        "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd",
        "manifest['batch_size'] == 1",
        "manifest['attn_implementation'] == 'eager'",
        "integrity['status'] == 'verified'",
    )
    if any(value not in command for value in required):
        raise ValueError("Command drift")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
    )
    if any(value in command.lower() for value in forbidden):
        raise ValueError("Secret/local/forbidden path drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
