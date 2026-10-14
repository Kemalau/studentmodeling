#!/usr/bin/env python3
"""Fail-closed validator for full recurrent M4 Feeds jobs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
TRAIN_SCRIPT_SHA256 = "865c12219deb23cf9db145c797244969721fce80323050dcd987b33d96cbfb1b"
PROJECTIONS = ("identity", "trainable")
ARMS = ("generation_only", "generation_kd")
RUN_KINDS = ("diagnostic", "full")
CACHE_JOB = "m4-full-teacher-cache-seed42"
EXPECTED_JOBS = {CACHE_JOB: ("cache", "cache")}
for kind in RUN_KINDS:
    for arm in ARMS:
        for projection in PROJECTIONS:
            EXPECTED_JOBS[
                f"m4-{kind}-{arm.replace('_', '-')}-{projection}-seed42"
            ] = (kind, arm, projection)
EXPECTED_JOBS["m4-memory-diagnostic-generation-kd-trainable-seed42"] = (
    "diagnostic",
    "generation_kd",
    "trainable",
)


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    candidates = [
        index
        for marker in ("description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not candidates:
        raise ValueError(f"No AMLT YAML found in {path}")
    config = yaml.safe_load(text[min(candidates) :])
    if not isinstance(config, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return config


def selected_job_name(kind: str, arm: str, projection: str) -> str:
    if kind not in RUN_KINDS or arm not in ARMS or projection not in PROJECTIONS:
        raise ValueError(f"Unknown M4 selection: {kind}/{arm}/{projection}")
    return f"m4-{kind}-{arm.replace('_', '-')}-{projection}-seed42"


def validate(
    config: dict,
    *,
    selected_names: set[str] | None = None,
) -> None:
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
    names = {job.get("name") for job in jobs}
    expected = selected_names if selected_names is not None else set(EXPECTED_JOBS)
    if names != expected or len(jobs) != len(expected):
        raise ValueError(f"Job set drift: expected={sorted(expected)} got={sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        if name not in EXPECTED_JOBS:
            raise ValueError(f"Unknown job: {name}")
        if job.get("sla_tier") != "Standard":
            raise ValueError(f"SLA drift in {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
            raise ValueError(f"UAI drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if name == CACHE_JOB:
            if job.get("sku") != "80G1-A100":
                raise ValueError(f"Cache GPU drift in {name}")
            required = (
                "teacher_cache_train_n12000_top256_fixedu4_v2_20260730",
                "--top-k 256",
                "assert len(cache.by_example) == 12000",
                "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            )
        else:
            kind, arm, projection = EXPECTED_JOBS[name]
            if kind == "full":
                if job.get("sku") != "80G4-A100" or job.get("mpi") is not True:
                    raise ValueError(f"Full GPU/MPI drift in {name}")
                if int(job.get("process_count_per_node", 0)) != -1:
                    raise ValueError(f"Full process-count drift in {name}")
            elif job.get("sku") != "80G1-A100":
                raise ValueError(f"Diagnostic GPU drift in {name}")
            if env.get("RUN_KIND") != kind or env.get("ARM") != arm or env.get(
                "PROJECTION_MODE"
            ) != projection:
                raise ValueError(f"M4 label drift in {name}")
            required = (
                "train_recurrent_student_m4.py",
                TRAIN_SCRIPT_SHA256,
                "--causal-weight 1.0",
                "--causal-margin 0.5",
                "18f2a05be56a93bb63f5ac552559502f29770158356faf906020911d5e9e2e6d",
                "edb1bda6cc6aceb4507ac599377284c533014cef34b80e1063e062a74752267e",
                "answer_anchor_visible_v2",
                "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            )
            if kind == "diagnostic" and any(
                value not in command
                for value in (
                    "--diagnostic-max-micro-steps",
                    "DIAGNOSTIC_STEPS",
                    "--eval-limit 40",
                    "echo 1 || echo 4",
                )
            ):
                raise ValueError(f"Diagnostic limit missing in {name}")
            if name.startswith("m4-memory-diagnostic-") and (
                env.get("DIAGNOSTIC_STEPS") != "200"
                or env.get("RUN_SUFFIX") != "sdpa_v1"
            ):
                raise ValueError(f"Memory diagnostic drift in {name}")
            if kind == "full" and any(
                value not in command
                for value in (
                    "--distributed-mpi",
                    "--distributed-world-size 4",
                    "--save-steps 250",
                    "m4-data-preparation.lock",
                    "m4-model-copy.lock",
                    "echo 1 || echo 4",
                )
            ):
                raise ValueError(f"Full distributed path missing in {name}")
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
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
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    selected = set(args.jobs) if args.jobs else None
    validate(load_config(args.config), selected_names=selected)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
