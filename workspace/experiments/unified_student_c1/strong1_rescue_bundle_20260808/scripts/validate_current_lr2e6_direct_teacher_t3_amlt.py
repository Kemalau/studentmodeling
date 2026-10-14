#!/usr/bin/env python3
"""Fail-closed preflight for the direct-teacher-state T3 diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "current-lr2e6-direct-teacher-t3-eedi75-seed42"


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
    if storage != {"storage_account_name": "csnewsandfeeds4150361735", "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True}:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job-set drift")
    job = jobs[0]
    if job.get("sku") != "80G1-A100" or job.get("mpi") is not False or int(job.get("process_count_per_node", 0)) != 1 or job.get("sla_tier") != "Standard":
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600 or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("Retry/duration/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    command = command.replace("$$", "$")
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models", "--state-architecture scalar", "eligible_for_final_gate=true")
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar/final-gate command drift")
    required = (
        "7214304189.17136-ef894bfd-a1be/model",
        "41c68adcf291c06cf4c1f776ff3b681e89421504aff437bd20b149393062c54e",
        "070c7e487cf721647278382c761d5eaa7861e2be72084d94db0a87f9a88220b7",
        "7bdb9b7b6e9f4c37db4c8551bba8eb9260a989e707fe535ff6922d03fe082cd2",
        "42bb943076e0d8a6a392db447021662b32ecd3c1691c7994a32788d8c7b2bb33",
        "evaluate_direct_teacher_t3.py",
        "--sources eedi_mining",
        "--max-samples-per-group 75",
        "diagnostic_only",
        "eligible_for_final_gate=false",
        "raw_history_visible_to_state_encoder",
        "raw_history_visible_to_decoder",
        "optimizer_steps=0",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Direct-teacher T3 command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
