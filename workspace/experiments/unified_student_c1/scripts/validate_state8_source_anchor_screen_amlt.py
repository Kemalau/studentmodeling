#!/usr/bin/env python3
"""Fail-closed AMLT validation for the source-aware State8 screen."""

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
RUNNER = "run_state8_source_anchor_screen.sh"
EXPECTED_JOBS = {
    "state8-source-anchor-full-mse1-seed3071": "full_mse1",
    "state8-source-anchor-updater-mse1-seed3072": "updater_mse1",
    "state8-source-anchor-full-mse2-seed3073": "full_mse2",
}
ADAPTER = "73b6633759febc6ccddac8e81d8715b512a86c3410cf6d2f160b48610d59c558"
STATE = "05dd9377584301293ba319ef7cd3bc93ec64f820c171f51abee593185a495aba"


def validate(path: Path, selected_jobs: set[str] | None = None) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
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
    code = config.get("code") or {}
    if not (code.get("local_dir") == "$CONFIG_DIR/.." or code.get("storage_id") == "default"):
        raise ValueError("Code root drift")
    jobs = config.get("jobs") or []
    expected_names = set(EXPECTED_JOBS) if selected_jobs is None else set(selected_jobs)
    if not expected_names or not expected_names <= set(EXPECTED_JOBS):
        raise ValueError(f"Invalid selected jobs: {sorted(expected_names)}")
    if {str(job.get("name")) for job in jobs} != expected_names:
        raise ValueError("Job matrix drift")
    runner_hash = hashlib.sha256(
        (Path(__file__).resolve().parent / RUNNER).read_bytes()
    ).hexdigest()
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku"),
            job.get("process_count_per_node"),
            job.get("mpi"),
            job.get("sla_tier"),
        ) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        env = submit.get("env") or {}
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-source-anchor-venv",
            "SOURCE_ANCHOR_MODE": EXPECTED_JOBS[name],
            "EXPECTED_STUDENT_ADAPTER_SHA256": ADAPTER,
            "EXPECTED_STUDENT_STATE_SHA256": STATE,
        }
        if env != expected:
            raise ValueError(f"Environment/artifact drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if runner_hash not in command or f"bash scripts/{RUNNER}" not in command:
            raise ValueError(f"Runner binding drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", action="append")
    args = parser.parse_args()
    validate(args.config, set(args.job) if args.job else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
