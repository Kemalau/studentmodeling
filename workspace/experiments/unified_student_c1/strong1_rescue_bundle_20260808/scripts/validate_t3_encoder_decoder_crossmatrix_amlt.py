#!/usr/bin/env python3
"""Fail-closed preflight for the missing T3 encoder/decoder matrix cell."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "current-lr2e6-fulltask-crossadapter-full-t3-seed42"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
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
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job-set drift")
    job = jobs[0]
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
        or submit.get("max_run_duration_seconds") != 43200
        or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
    ):
        raise ValueError("Retry/duration/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    command = command.replace("$$", "$")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--state-architecture scalar",
        "--max-samples-per-group",
    )
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar/partial path leaked")
    required = (
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "7214304189.17136-ef894bfd-a1be/model",
        "qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter",
        "41c68adcf291c06cf4c1f776ff3b681e89421504aff437bd20b149393062c54e",
        "070c7e487cf721647278382c761d5eaa7861e2be72084d94db0a87f9a88220b7",
        "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a",
        "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
        "--recurrent-state",
        "--decode-adapter",
        "--state-architecture layerwise",
        "--tasks diagnose_misconception",
        "observed==1494",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "optimizer_steps=0",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Cross-matrix command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
