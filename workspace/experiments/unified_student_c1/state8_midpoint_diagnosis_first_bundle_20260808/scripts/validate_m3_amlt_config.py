#!/usr/bin/env python3
"""Fail-closed validation for the unified recurrent M3 Feeds config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
RECURRENT_MODEL_SHA256 = "9e0431dccb524f67b401f7ea911e11c96fc3061432ffdc73e0cac2bbc2acc5a0"
ARMS = ("generation_only", "generation_kd")
PROJECTIONS = ("identity", "trainable")
EXPECTED_JOBS = {
    f"m3-{arm.replace('_', '-')}-{projection}-seed42": (arm, projection)
    for projection in PROJECTIONS
    for arm in ARMS
}


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
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


def selected_names(projection: str) -> set[str]:
    if projection not in PROJECTIONS:
        raise ValueError(f"Unknown projection mode: {projection}")
    return {
        name
        for name, (_, job_projection) in EXPECTED_JOBS.items()
        if job_projection == projection
    }


def validate(
    config: dict,
    *,
    projection: str | None = None,
    selected_only: bool = False,
) -> None:
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
    actual_names = {job.get("name") for job in jobs}
    expected_names = (
        selected_names(projection)
        if selected_only and projection is not None
        else set(EXPECTED_JOBS)
    )
    if actual_names != expected_names or len(jobs) != len(expected_names):
        raise ValueError(
            f"Job set drift: expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )
    if projection is not None:
        disallowed = {
            name
            for name in actual_names
            if EXPECTED_JOBS[name][1] != projection
        }
        if disallowed:
            raise ValueError(f"Projection selection drift: {sorted(disallowed)}")

    for job in jobs:
        name = job["name"]
        arm, job_projection = EXPECTED_JOBS[name]
        if job.get("sku") != "80G1-A100" or job.get("sla_tier") != "Standard":
            raise ValueError(f"GPU/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
            raise ValueError(f"UAI drift in {name}")
        if env.get("ARM") != arm or env.get("PROJECTION_MODE") != job_projection:
            raise ValueError(f"M3 label drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            RECURRENT_MODEL_SHA256,
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "/mnt/amulet/studentmodeling/method_small_v1",
            "/mnt/amulet/studentmodeling/adapters/qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter",
            "/mnt/amulet/studentmodeling/world_model_v1/m3/teacher_cache_train_n1200_top256_fixedu4_v2_20260730",
            "recurrent_malrule_cf1_u4_",
            "--max-steps 1200",
            "--gradient-accumulation-steps 4",
            "--eval-limit 300",
            "build_teacher_cache.py",
            "expected_source_sha256='6fa105b8635e19fb0acf9f9f145d3fe5f8bd9715927f891d647b298ffb920bc1'",
            "completion.json",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Training command drift in {name}")
        if arm == "generation_kd" and "--behavior-kd-weight 0.5" not in command:
            raise ValueError(f"KD command missing in {name}")
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Secret or local-model path in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS)
    parser.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    if args.selected_only and args.projection is None:
        parser.error("--selected-only requires --projection")
    validate(
        load_config(args.config),
        projection=args.projection,
        selected_only=args.selected_only,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
