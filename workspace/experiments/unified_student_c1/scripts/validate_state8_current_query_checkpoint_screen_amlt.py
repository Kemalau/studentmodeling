#!/usr/bin/env python3
"""Fail-closed AMLT validation for a bound query-aligned checkpoint screen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
RUNNER = "run_state8_current_query_checkpoint_screen.sh"
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])


def valid_hash(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate(path: Path, expected_step: int | None = None) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|HASH_PENDING|RUNNER_SHA256", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    if (config.get("code") or {}).get("local_dir") != PROJECT_ROOT:
        raise ValueError("Code root drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Exactly one bound milestone is required")
    job = jobs[0]
    env = (job.get("submit_args") or {}).get("env") or {}
    step = int(env.get("CHECKPOINT_STEP", -1))
    if step not in {20, 100} or (expected_step is not None and step != expected_step):
        raise ValueError("Checkpoint step drift")
    if job.get("name") != f"state8-current-query-cp{step}-screen":
        raise ValueError("Job/step drift")
    if (
        job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
        raise ValueError("Retry/duration drift")
    fixed = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-current-query-screen-venv",
    }
    if any(env.get(key) != value for key, value in fixed.items()):
        raise ValueError("Identity/environment drift")
    for key in (
        "CHECKPOINT_ADAPTER_SHA256", "CHECKPOINT_STATE_SHA256", "CHECKPOINT_RECEIPT_SHA256"
    ):
        if not valid_hash(env.get(key)):
            raise ValueError(f"Invalid checkpoint binding: {key}")
    command = command_text(job.get("command")); validate_shell_syntax(command, str(job["name"]))
    runner_hash = hashlib.sha256((Path(__file__).resolve().parent / RUNNER).read_bytes()).hexdigest()
    if runner_hash not in command or f"bash scripts/{RUNNER}" not in command:
        raise ValueError("Runner binding drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    validate(args.config, args.expected_step)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
