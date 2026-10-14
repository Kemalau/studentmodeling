#!/usr/bin/env python3
"""Fail-closed validator for the exact known-OOM-row offload diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-known-oom-row-recovery"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
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
    expected_storage = {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }
    if storage != expected_storage:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if (
        submit.get("max_attempts") != 1
        or submit.get("max_run_duration_seconds") != 21600
        or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
    ):
        raise ValueError("Retry/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
        "cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py",
        "cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py",
        "--state-architecture layerwise --layerwise-operation-checkpointing",
        "--layerwise-transition-rank 256",
        "--max-length 1536 --direct-offload-min-tokens 512",
        "--diagnostic-max-micro-steps 1 --diagnostic-real-index 177752",
        "history[0]['direct_offload_used'] is True",
        "history[0]['direct_tokens'] == 549",
        "history[0]['peak_cuda_allocated_gib'] < 79.0",
        "history[0]['peak_cuda_reserved_gib'] < 79.0",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Known-OOM-row command drift")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--distributed-mpi",
    )
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local/distributed path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
