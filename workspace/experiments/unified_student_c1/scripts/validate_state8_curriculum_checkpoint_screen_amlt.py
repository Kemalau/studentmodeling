#!/usr/bin/env python3
"""Fail-closed validation for a bound curriculum checkpoint screen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from materialize_state8_curriculum_checkpoint_screen import ARMS, STEPS
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )
    from scripts.materialize_state8_curriculum_checkpoint_screen import ARMS, STEPS


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
RUNNER = "run_state8_curriculum_checkpoint_screen.sh"
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
AMLT_DUMP_VERSION = "11.14.2"
AMLT_CODE_STORAGE = {
    "storage_account_name": "csnewsandfeeds4150361735",
    "container_name": "amulet",
    "mount_dir": "/mnt/default",
    "is_output": True,
}


def valid_hash(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_code_root(config: dict) -> None:
    """Accept source and AMLT-dump representations of the same code root.

    AMLT 11.14.2 removes ``code.local_dir`` after staging the configured
    project and replaces it with ``code.storage_id: default``.  Treat that
    normalized representation as equivalent only when its version and exact
    backing storage binding match the project.  This remains fail-closed for
    an omitted/foreign code root.
    """
    code = config.get("code") or {}
    local_dir = code.get("local_dir")
    if local_dir is not None:
        if str(Path(str(local_dir)).resolve()) != PROJECT_ROOT:
            raise ValueError("Code root drift")
        return
    storage = config.get("storage") or {}
    if (
        str(config.get("version") or "") != AMLT_DUMP_VERSION
        or code.get("storage_id") != "default"
        or storage.get("default") != AMLT_CODE_STORAGE
    ):
        raise ValueError("Code root drift")


def validate(
    path: Path,
    expected_arm: str | None = None,
    expected_step: int | None = None,
) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|\bREQUIRED\b|PENDING|RUNNER_SHA256", rendered):
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
    validate_code_root(config)
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Exactly one bound checkpoint is required")
    job = jobs[0]
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    family = str(env.get("CHECKPOINT_FAMILY") or "")
    arm = str(env.get("CURRICULUM_ARM") or "")
    step = int(env.get("CHECKPOINT_STEP", -1))
    if family != "curriculum":
        raise ValueError("Curriculum checkpoint family drift")
    if arm not in ARMS or step not in STEPS:
        raise ValueError("Curriculum arm/step drift")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("Expected curriculum arm drift")
    if expected_step is not None and step != expected_step:
        raise ValueError("Expected checkpoint step drift")
    if job.get("name") != f"state8-curriculum-{arm.replace('_', '-')}-cp{step}-screen":
        raise ValueError("Job/arm/step drift")
    expected_root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        f"{ARMS[arm]}/checkpoint-{step}"
    )
    if env.get("CHECKPOINT_ROOT") != expected_root:
        raise ValueError("Checkpoint root drift")
    if (
        job.get("sku"),
        job.get("process_count_per_node"),
        job.get("mpi"),
        job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
        raise ValueError("Retry/duration drift")
    fixed = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-curriculum-screen-venv",
    }
    if any(env.get(key) != value for key, value in fixed.items()):
        raise ValueError("Identity/environment drift")
    for key in (
        "CHECKPOINT_ADAPTER_SHA256",
        "CHECKPOINT_STATE_SHA256",
        "CHECKPOINT_RECEIPT_SHA256",
    ):
        if not valid_hash(env.get(key)):
            raise ValueError(f"Invalid checkpoint binding: {key}")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, str(job["name"]))
    runner_hash = hashlib.sha256(
        (Path(__file__).resolve().parent / RUNNER).read_bytes()
    ).hexdigest()
    if runner_hash not in command or f"bash scripts/{RUNNER}" not in command:
        raise ValueError("Runner binding drift")
    if any(fragment in rendered for fragment in ("optimizer.step()", "--init-adapter")):
        raise ValueError("Read-only screen mutates weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-arm", choices=sorted(ARMS))
    parser.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    validate(args.config, args.expected_arm, args.expected_step)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
