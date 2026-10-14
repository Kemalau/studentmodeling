#!/usr/bin/env python3
"""Fail-closed AMLT validation for the independent full-history teacher."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-full-history-natural-teacher-seed42"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
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
    code = config.get("code") or {}
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job-set drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("mpi") is not False
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("UAI drift")
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
        raise ValueError("Retry/duration drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models")
    if any(item.lower() in command.lower() for item in forbidden):
        raise ValueError("Secret or local model path leaked")
    for fragment in (
        "FAILED line=%s rc=%s command=%s",
        "build_state8_teacher_views.py",
        "qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter",
        "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a",
        "--max-steps -1 --num-train-epochs 1",
        "--max-train-samples 6476 --max-valid-samples 300",
        "--gradient-accumulation-steps 16",
        "--init-adapter",
        "state8_full_history_teacher_seed42_v1_20260805",
        "latest_checkpoint",
        "--resume-from-checkpoint",
        "--save-steps 100 --save-total-limit 2",
        "7c8b8b0a32ecf5695b7200f2c5b64fb0989598b88dff1da59da7cbfe7752d591",
        "'train_eval':0",
        "o['eval']['rows']==1494",
        "eedi_mining':894",
    ):
        if fragment not in command:
            raise ValueError(f"Teacher launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
