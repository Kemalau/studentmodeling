#!/usr/bin/env python3
"""Fail-closed preflight for the two-cell direct-teacher T3 cross-matrix."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
EXPECTED = {
    "current-teacher-full-decoder-eedi75-seed42": {
        "condition": "current_teacher_full_decoder",
        "capture": "/mnt/amulet/projects/v-yuejiali/amlt-results/7214304189.17136-ef894bfd-a1be/model",
        "capture_sha": "41c68adcf291c06cf4c1f776ff3b681e89421504aff437bd20b149393062c54e",
        "state": "/mnt/amulet/projects/v-yuejiali/amlt-results/7214304189.17136-ef894bfd-a1be/model/recurrent_state.pt",
        "state_sha": "070c7e487cf721647278382c761d5eaa7861e2be72084d94db0a87f9a88220b7",
    },
    "fulltask-teacher-full-decoder-eedi75-seed42": {
        "condition": "fulltask_teacher_full_decoder",
        "capture": "/mnt/amulet/studentmodeling/adapters/qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter",
        "capture_sha": "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a",
        "state": "none",
        "state_sha": "none",
    },
}
DECODE = "/mnt/amulet/studentmodeling/adapters/qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


def validate(config: dict, selected_jobs: set[str] | None = None) -> None:
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
    names = {str(job.get("name")) for job in jobs}
    expected_names = selected_jobs or set(EXPECTED)
    if names != expected_names or not names <= set(EXPECTED):
        raise ValueError(f"Job-set drift: {names}")
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        expected = EXPECTED[name]
        if (
            submit.get("max_attempts") != 2
            or submit.get("max_run_duration_seconds") != 21600
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("CONDITION") != expected["condition"]
            or env.get("CAPTURE_ADAPTER") != expected["capture"]
            or env.get("CAPTURE_SHA256") != expected["capture_sha"]
            or env.get("RECURRENT_STATE") != expected["state"]
            or env.get("RECURRENT_STATE_SHA256") != expected["state_sha"]
            or env.get("DECODE_ADAPTER") != DECODE
        ):
            raise ValueError(f"Retry/UAI/artifact drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        command = command.replace("$$", "$")
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "--state-architecture scalar",
            "eligible_for_final_gate=true",
        )
        if any(fragment.lower() in command.lower() for fragment in forbidden):
            raise ValueError(f"Secret/local/scalar/final-gate drift: {name}")
        required = (
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "bff7726953ae048bedf6dacdbb7500a54cb8de305e406f2497ef1f11c04167f8",
            "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
            "evaluate_direct_teacher_t3.py",
            "--decode-adapter",
            "--state-architecture layerwise",
            "--sources eedi_mining",
            "--max-samples-per-group 75",
            "raw_history_visible_to_state_encoder",
            "raw_history_visible_to_decoder",
            "optimizer_steps=0",
        )
        if any(fragment not in command for fragment in required):
            raise ValueError(f"Direct-teacher command drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
