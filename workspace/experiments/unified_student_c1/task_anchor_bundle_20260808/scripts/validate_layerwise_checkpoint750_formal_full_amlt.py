#!/usr/bin/env python3
"""Fail-closed validator for round-750 full-support T2/T3/T5 inference."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
TASKS = {
    "layerwise-checkpoint750-full-t2-seed42": "predict_response",
    "layerwise-checkpoint750-full-t3-seed42": "diagnose_misconception",
    "layerwise-checkpoint750-full-t5-seed42": "predict_prospective_error",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
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
    expected = set(TASKS) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(TASKS):
        raise ValueError("Expected job-set drift")
    jobs = config.get("jobs") or []
    if len(jobs) != len(expected) or {job.get("name") for job in jobs} != expected:
        raise ValueError("Job-set drift")
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError("GPU/MPI/SLA drift")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or submit.get("max_run_duration_seconds") != 259200
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("TASKS") != TASKS[name]
        ):
            raise ValueError("Retry/duration/UAI/task drift")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            "checkpoint-round-750",
            "receipt.get('round')==750",
            "receipt.get('world_size')==16",
            "receipt['sha256']['adapter_model']",
            "receipt['sha256']['recurrent_state']",
            "--state-architecture layerwise",
            "raw_history_visible_to_decoder=false",
            "persistent_history_kv_cache=false",
            "stage=layerwise_finalfit_checkpoint750_formal_full",
            "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
            "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
            "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        )
        if any(fragment not in command for fragment in required):
            raise ValueError("Full-support command drift")
        forbidden = (
            "--max-samples-per-group",
            "checkpoint-round-1000",
            "final_fit/seed42.json",
            "final_adapter",
            "--state-architecture scalar",
            "/home/v-yuejiali/studentmodeling/models",
            "password",
            "account_key",
            "access_token",
        )
        if any(fragment.lower() in command.lower() for fragment in forbidden):
            raise ValueError("Small/final/local/scalar/secret path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
