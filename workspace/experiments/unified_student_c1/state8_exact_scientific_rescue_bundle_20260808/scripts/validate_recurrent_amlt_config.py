#!/usr/bin/env python3
"""Fail-closed validation for the frozen recurrent M2 Feeds config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches


EXPECTED_JOBS = {
    "recurrent-u1-identity-seed42": ("1", "identity"),
    "recurrent-u1-trainable-seed42": ("1", "trainable"),
    "recurrent-u4-identity-seed42": ("4", "identity"),
    "recurrent-u4-trainable-seed42": ("4", "trainable"),
}
IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
RECURRENT_MODEL_SHA256 = "9e0431dccb524f67b401f7ea911e11c96fc3061432ffdc73e0cac2bbc2acc5a0"


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    # AMLT --dump may prefix informational lines.  The actual normalized YAML
    # begins at description or target.
    candidates = [
        index
        for marker in ("description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not candidates:
        raise ValueError(f"No AMLT YAML document found in {path}")
    config = yaml.safe_load(text[min(candidates) :])
    if not isinstance(config, dict):
        raise ValueError(f"Invalid AMLT config in {path}")
    return config


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder in AMLT config")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden container image")
    target = config.get("target") or {}
    if not target_matches(target, WORKSPACE):
        raise ValueError(f"Target drift: {target}")
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
        raise ValueError(f"Storage drift: {storage}")
    jobs = config.get("jobs") or []
    if {job.get("name") for job in jobs} != set(EXPECTED_JOBS):
        raise ValueError("Job set drift")
    if len(jobs) != 4:
        raise ValueError("Expected exactly four independent jobs")
    for job in jobs:
        name = job["name"]
        unroll, projection = EXPECTED_JOBS[name]
        if job.get("sku") != "80G1-A100" or job.get("sla_tier") != "Standard":
            raise ValueError(f"GPU/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
            raise ValueError(f"UAI drift in {name}")
        if str(env.get("UNROLL")) != unroll or env.get("PROJECTION_MODE") != projection:
            raise ValueError(f"Factorial label drift in {name}")
        command = command_text(job.get("command"))
        required = (
            RECURRENT_MODEL_SHA256,
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "/mnt/amulet/studentmodeling/state_curriculum_stage_a_v1",
            "/mnt/amulet/studentmodeling/checkpoints/state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727/final_adapter",
            "--max-steps 400",
            "--gradient-accumulation-steps 4",
            "--eval-limit 100",
            "completion.json",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Training command drift in {name}")
        forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models")
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Secret or local-model path in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load_config(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
