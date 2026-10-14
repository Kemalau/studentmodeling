#!/usr/bin/env python3
"""Fail-closed validator for the zero-training layerwise formal screen."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_layerwise_formal_screen import (
    CAPTURE_ADAPTER,
    DECODE_ADAPTER,
    EVALUATOR_SHA256,
    EVALUATOR_TEST_SHA256,
    LAYERWISE_SHA256,
    LAYERWISE_TEST_SHA256,
    SPECS,
)


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [
        index
        for marker in ("description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
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
    expected = {f"layerwise-formal-screen-{group}-seed42": (group, tasks) for group, tasks in SPECS.items()}
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != set(expected) or len(jobs) != 4:
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        group, tasks = expected[name]
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            job.get("sku") != "80G1-A100"
            or job.get("sla_tier") != "Standard"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or submit.get("max_attempts") != 2
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("EVAL_GROUP") != group
            or env.get("TASKS") != tasks
        ):
            raise ValueError(f"GPU/UAI/task drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            CAPTURE_ADAPTER,
            DECODE_ADAPTER,
            EVALUATOR_SHA256,
            LAYERWISE_SHA256,
            EVALUATOR_TEST_SHA256,
            LAYERWISE_TEST_SHA256,
            "--state-architecture layerwise",
            "--decode-adapter",
            "--max-samples-per-group 50",
            "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
            "raw_history_visible_to_decoder=false",
            "persistent_history_kv_cache=false",
            "optimizer_steps=0",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        )
        forbidden = (
            "train_recurrent_student",
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Training/secret/local path leaked into {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
