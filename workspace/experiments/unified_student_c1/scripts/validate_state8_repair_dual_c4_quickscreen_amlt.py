#!/usr/bin/env python3
"""Fail-closed validation for the causal-teacher cp20 dual-C4 quickscreen."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from validate_state8_curriculum_checkpoint_screen_amlt import (
        IMAGE,
        WORKSPACE,
        valid_hash,
        validate_code_root,
    )
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from scripts.validate_state8_curriculum_checkpoint_screen_amlt import (
        IMAGE,
        WORKSPACE,
        valid_hash,
        validate_code_root,
    )


RUNNER = "run_state8_repair_dual_c4_quickscreen.sh"
SPECS = {(8201, "causal_teacher_low"), (8202, "causal_teacher_high")}


def validate(path: Path) -> None:
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
    if len(jobs) != 2:
        raise ValueError("Two independent one-GPU quickscreen jobs are required")
    observed = set()
    runner_sha = hashlib.sha256((Path(__file__).resolve().parent / RUNNER).read_bytes()).hexdigest()
    for job in jobs:
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError("Retry/duration drift")
        env = submit.get("env") or {}
        seed = int(env.get("REPAIR_SEED", -1))
        branch = str(env.get("REPAIR_BRANCH") or "")
        coordinate = (seed, branch)
        if coordinate not in SPECS or coordinate in observed:
            raise ValueError("Repair coordinate drift")
        observed.add(coordinate)
        expected_name = f"state8-ct-{branch.replace('_', '-')}-seed{seed}-cp20-dual-c4"
        if job.get("name") != expected_name:
            raise ValueError("Job name drift")
        fixed = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-repair-dual-c4-venv",
            "CHECKPOINT_STEP": "20",
            "CHECKPOINT_ROOT": (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                "state8_curriculum_repairs_20260812/"
                f"seed{seed}_{branch}/checkpoint-20"
            ),
        }
        if any(str(env.get(key)) != value for key, value in fixed.items()):
            raise ValueError("Identity/environment/checkpoint drift")
        for key in ("CHECKPOINT_ADAPTER_SHA256", "CHECKPOINT_STATE_SHA256", "CHECKPOINT_RECEIPT_SHA256"):
            if not valid_hash(env.get(key)):
                raise ValueError(f"Invalid hash: {key}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, expected_name)
        if runner_sha not in command or f"bash scripts/{RUNNER}" not in command:
            raise ValueError("Runner binding drift")
    if observed != SPECS:
        raise ValueError("Quickscreen matrix drift")
    if any(fragment in rendered for fragment in ("optimizer.step()", "--init-adapter")):
        raise ValueError("Read-only quickscreen mutates weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
