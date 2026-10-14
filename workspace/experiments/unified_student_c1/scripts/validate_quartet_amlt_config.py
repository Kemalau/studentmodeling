#!/usr/bin/env python3
"""Fail-closed validation for the frozen quartet-v2 Feeds config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
EXPECTED = {
    "quartet-control-seed42": ("control", "all"),
    "quartet-swap-all-seed42": ("paired_swap", "all"),
    "quartet-swap-latehalf-seed42": (
        "paired_swap",
        ",".join(str(index) for index in range(32, 64)),
    ),
}


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [index for marker in ("description:", "target:") if (index := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    target = config.get("target") or {}
    if not target_matches(target, WORKSPACE):
        raise ValueError(f"Target drift: {target}")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError(f"Storage drift: {storage}")
    jobs = config.get("jobs") or []
    if len(jobs) != 3 or {job.get("name") for job in jobs} != set(EXPECTED):
        raise ValueError("Quartet job set drift")
    for job in jobs:
        name = job["name"]
        arm, layers = EXPECTED[name]
        if job.get("sku") != "80G1-A100" or job.get("sla_tier") != "Standard":
            raise ValueError(f"GPU/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
            raise ValueError(f"UAI drift in {name}")
        if env.get("ARM") != arm or str(env.get("PATCH_LAYERS")) != layers:
            raise ValueError(f"Arm/layer drift in {name}")
        command = command_text(job.get("command"))
        required = (
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean",
            "/mnt/amulet/studentmodeling/state_curriculum_stage_a_v1",
            "fc329808f54edff7c12586ed9467801dd287efdf587599f5f456548d3979140e",
            "--max-steps 500",
            "--resume-from-checkpoint",
            "gate_summary.json",
        )
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
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
