#!/usr/bin/env python3
"""Fail-closed AMLT validation for a bound repair checkpoint screen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from audit_state8_curriculum_repair_checkpoint import SPECS, STEPS
    from validate_state8_curriculum_checkpoint_screen_amlt import (
        AMLT_CODE_STORAGE,
        AMLT_DUMP_VERSION,
        IMAGE,
        WORKSPACE,
        valid_hash,
        validate_code_root,
    )
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )
    from scripts.audit_state8_curriculum_repair_checkpoint import SPECS, STEPS
    from scripts.validate_state8_curriculum_checkpoint_screen_amlt import (
        AMLT_CODE_STORAGE,
        AMLT_DUMP_VERSION,
        IMAGE,
        WORKSPACE,
        valid_hash,
        validate_code_root,
    )


RUNNER = "run_state8_curriculum_checkpoint_screen.sh"


def validate(
    path: Path,
    expected_seed: int | None = None,
    expected_branch: str | None = None,
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
        raise ValueError("Exactly one bound repair checkpoint is required")
    job = jobs[0]
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    seed = int(env.get("REPAIR_SEED", -1))
    branch = str(env.get("REPAIR_BRANCH") or "")
    step = int(env.get("CHECKPOINT_STEP", -1))
    if env.get("CHECKPOINT_FAMILY") != "repair":
        raise ValueError("Repair checkpoint family drift")
    if (seed, branch) not in SPECS or step not in STEPS:
        raise ValueError("Repair coordinate drift")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("Expected repair seed drift")
    if expected_branch is not None and branch != expected_branch:
        raise ValueError("Expected repair branch drift")
    if expected_step is not None and step != expected_step:
        raise ValueError("Expected repair step drift")
    arm = f"repair_{branch}_seed{seed}"
    if env.get("CURRICULUM_ARM") != arm:
        raise ValueError("Repair arm drift")
    expected_name = (
        f"state8-repair-{branch.replace('_', '-')}-seed{seed}-cp{step}-screen"
    )
    if job.get("name") != expected_name:
        raise ValueError("Repair job coordinate drift")
    expected_root = (
        "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
        "state8_curriculum_repairs_20260812/"
        f"seed{seed}_{branch}/checkpoint-{step}"
    )
    if env.get("CHECKPOINT_ROOT") != expected_root:
        raise ValueError("Repair checkpoint root drift")
    if (
        job.get("sku"),
        job.get("process_count_per_node"),
        job.get("mpi"),
        job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
        raise ValueError("Retry/duration drift")
    if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("Identity drift")
    if env.get("VENV_ROOT") != "/tmp/studentmodeling-state8-curriculum-screen-venv":
        raise ValueError("Environment drift")
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
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-branch")
    parser.add_argument("--expected-step", type=int)
    args = parser.parse_args()
    validate(
        args.config,
        args.expected_seed,
        args.expected_branch,
        args.expected_step,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
