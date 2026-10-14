#!/usr/bin/env python3
"""Fail-closed validation for the selected State8 answer-C4 AMLT config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ARMS = {"full_mse1", "updater_mse1", "full_mse2"}
JOBS = {
    "state8-source-anchor-full-mse1-seed3071",
    "state8-source-anchor-updater-mse1-seed3072",
    "state8-source-anchor-full-mse2-seed3073",
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text())
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"RUNNER_SHA256|SELECTED_|<placeholder>|\bREQUIRED\b", rendered):
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
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != "state8-source-anchor-answer-c4-selected":
        raise ValueError("Job drift")
    job = jobs[0]
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
        raise ValueError("Retry/duration drift")
    env = submit.get("env") or {}
    if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("UAI drift")
    if env.get("SOURCE_ANCHOR_ARM") not in ARMS or env.get("SOURCE_ANCHOR_JOB") not in JOBS:
        raise ValueError("Selected arm/job drift")
    for key in ("SOURCE_ANCHOR_ADAPTER_SHA256", "SOURCE_ANCHOR_STATE_SHA256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(env.get(key) or "")):
            raise ValueError(f"Invalid artifact hash: {key}")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, str(job["name"]))
    if "bash scripts/run_state8_source_anchor_answer_c4.sh" not in command:
        raise ValueError("Runner drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
