#!/usr/bin/env python3
"""Fail-closed Feeds preflight for the five task-anchor milestone screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
STEPS = (200, 400, 600, 800, 1000)
RUNNER_SHA = "520da36d9fe23f35712a490c90aba36ce59ce2220becdea712f14b609e7a04f0"
AUDIT_SHA = "363316b2b7d49af0f5448666fab0592acbae42dc1214a0ca164bc7c226da210b"


def validate(path: Path, *, allow_job_subset: bool = False) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
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
    expected_names = {f"state8-taskanchor-milestone-{step}" for step in STEPS}
    observed_names = [str(job.get("name")) for job in jobs]
    observed_name_set = set(observed_names)
    valid_job_set = (
        bool(jobs)
        and len(observed_names) == len(observed_name_set)
        and (
            observed_name_set <= expected_names
            if allow_job_subset
            else observed_name_set == expected_names
            and len(jobs) == len(STEPS)
        )
    )
    if not valid_job_set:
        raise ValueError("Job set drift")
    root = Path(__file__).resolve().parents[1]
    observed_runner = hashlib.sha256((root / "scripts/run_state8_taskanchor_milestone_screen.sh").read_bytes()).hexdigest()
    observed_audit = hashlib.sha256((root / "scripts/audit_state8_taskanchor_milestone.py").read_bytes()).hexdigest()
    if (observed_runner, observed_audit) != (RUNNER_SHA, AUDIT_SHA):
        raise ValueError("Local source hash drift")
    for job in jobs:
        name = str(job["name"])
        step = int(name.rsplit("-", 1)[1])
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("VENV_ROOT") != "/tmp/studentmodeling-state8-taskanchor-milestone-venv":
            raise ValueError(f"Venv drift: {name}")
        if env.get("MILESTONE_STEP") != str(step):
            raise ValueError(f"Step drift: {name}")
        receipt = Path(f"/tmp/state8-taskanchor-milestone-receipts/{step}.json")
        if receipt.is_file():
            value = json.loads(receipt.read_text(encoding="utf-8"))
            expected = {
                "MILESTONE_ADAPTER_SHA256": value["sha256"]["adapter_model"],
                "MILESTONE_STATE_SHA256": value["sha256"]["recurrent_state"],
                "MILESTONE_RECEIPT_SHA256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            }
            if any(env.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Artifact binding drift: {name}")
        for key in ("MILESTONE_ADAPTER_SHA256", "MILESTONE_STATE_SHA256", "MILESTONE_RECEIPT_SHA256"):
            value = str(env.get(key, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Invalid digest: {name}/{key}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if RUNNER_SHA not in command or AUDIT_SHA not in command:
            raise ValueError(f"Command hash binding drift: {name}")
        if "run_state8_taskanchor_milestone_screen.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    if any(fragment in rendered for fragment in ("optimizer.step()", "--resume-from-checkpoint", "--init-adapter")):
        raise ValueError("Read-only screen mutates weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-job-subset",
        action="store_true",
        help=(
            "Allow a non-empty, duplicate-free subset of the five hash-bound "
            "jobs when validating an AMLT --dump rendered with selectors."
        ),
    )
    args = parser.parse_args()
    validate(args.config, allow_job_subset=args.allow_job_subset)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
