#!/usr/bin/env python3
"""Fail-closed validator for the exact 16-way final-fit teacher cache."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
ROOT = "/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way"
EXPECTED = {f"finalfit-cache-v2-rank{rank:02d}-seed42": rank for rank in range(16)}


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [index for marker in ("description:", "target:") if (index := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict, selected_names: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(
        r"<placeholder>|<workspace-name>|<your-command>|<describe your experiment>|"
        r"<job-name>|<file>|\bREQUIRED\b|PLACEHOLDER",
        rendered,
    ):
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
    names = {str(job.get("name")) for job in jobs}
    expected_names = set(EXPECTED) if selected_names is None else selected_names
    if not expected_names or not expected_names.issubset(EXPECTED):
        raise ValueError("Unknown selected cache job")
    if names != expected_names or len(jobs) != len(expected_names):
        raise ValueError("Job-set drift")
    for job in jobs:
        name = str(job["name"])
        rank = EXPECTED[name]
        if job.get("sku") != "80G1-A100" or job.get("mpi") is not False:
            raise ValueError(f"GPU/MPI drift in {name}")
        if int(job.get("process_count_per_node", 0)) != 1 or job.get("sla_tier") != "Standard":
            raise ValueError(f"Process/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 3 or int(submit.get("max_run_duration_seconds", 0)) != 345600:
            raise ValueError(f"Retry/duration drift in {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI or str(env.get("PARTITION_INDEX")) != str(rank):
            raise ValueError(f"Identity/partition drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            ROOT,
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "/mnt/amulet/studentmodeling/full_trainplusvalid_no_t4_strict_t6/train.jsonl",
            "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd",
            "d52b5f17ffbbd033ce2ba27877dbedf9abdaffa4aaaffe8194cbe7bd5a5b7f08",
            "d70a169b26f9e632c2104d90a2f4e07913350f4931424de3429c0a1ff922135e",
            "--partition-count 16",
            "--batch-size 1",
            "--attn-implementation eager",
            "--order-seed 42",
            "--top-k 256",
            "--shard-size 512",
            "manifest['rows'] in {16240, 16241}",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        lowered = command.lower()
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in lowered for value in forbidden):
            raise ValueError(f"Secret/local/forbidden path drift in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load_config(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
