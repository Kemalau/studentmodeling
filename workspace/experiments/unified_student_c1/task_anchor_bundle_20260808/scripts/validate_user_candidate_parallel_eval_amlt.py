#!/usr/bin/env python3
"""Fail-closed validation for the exact lr2e-6 candidate formal-eval jobs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "user-candidate-lr2e6-t2-seed42": ("t2", "predict_response", "3693"),
    "user-candidate-lr2e6-t3-seed42": ("t3", "diagnose_misconception", "1494"),
    "user-candidate-lr2e6-t5-seed42": ("t5", "predict_prospective_error", "4600"),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
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
    jobs = config.get("jobs") or []
    observed = {str(job.get("name")) for job in jobs}
    expected = set(JOBS) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(JOBS) or observed != expected:
        raise ValueError("Job-set drift")
    commands: list[str] = []
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or submit.get("max_run_duration_seconds") != 43200
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
        ):
            raise ValueError(f"Retry/duration/UAI drift: {name}")
        group, task, support = JOBS[name]
        if (
            str(env.get("EVAL_GROUP")),
            str(env.get("TASKS")),
            str(env.get("TASK_KEY")),
            str(env.get("EXPECTED_N")),
        ) != (group, task, task, support):
            raise ValueError(f"Task/support drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Candidate eval jobs do not share one controlled command")
    command = commands[0].replace("$$", "$")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--state-architecture scalar",
    )
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar path leaked")
    required = (
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "7214304189.17136-ef894bfd-a1be/model",
        "41c68adcf291c06cf4c1f776ff3b681e89421504aff437bd20b149393062c54e",
        "070c7e487cf721647278382c761d5eaa7861e2be72084d94db0a87f9a88220b7",
        "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
        "--state-architecture layerwise",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "--tasks $TASKS",
        "observed==int(sys.argv[3])",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Candidate formal-eval command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
