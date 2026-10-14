#!/usr/bin/env python3
"""Fail-closed validation for the three-arm State8 label-readout probe."""

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


RUNNER = "run_state8_eedi_label_readout.sh"
SPECS = {
    "early-label-low-cp20": (8301, "early_label_low", 20),
    "early-label-high-cp20": (8302, "early_label_high", 20),
    "causal-teacher-low-cp100-control": (8201, "causal_teacher_low", 100),
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b|PENDING|MATERIALIZATION", rendered):
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
    if len(jobs) != len(SPECS):
        raise ValueError("Three independent read-only jobs are required")
    runner_sha = hashlib.sha256(
        (Path(__file__).resolve().parent / RUNNER).read_bytes()
    ).hexdigest()
    observed = set()
    for job in jobs:
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 10800:
            raise ValueError("Retry/duration drift")
        env = submit.get("env") or {}
        tag = str(env.get("READOUT_TAG") or "")
        if tag not in SPECS or tag in observed:
            raise ValueError("Readout coordinate drift")
        observed.add(tag)
        seed, branch, step = SPECS[tag]
        fixed = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-label-readout-venv",
            "CHECKPOINT_ROOT": (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                f"state8_curriculum_repairs_20260812/seed{seed}_{branch}/checkpoint-{step}"
            ),
        }
        if any(str(env.get(key)) != value for key, value in fixed.items()):
            raise ValueError("Identity/checkpoint drift")
        for key in ("CHECKPOINT_ADAPTER_SHA256", "CHECKPOINT_STATE_SHA256"):
            if not valid_hash(env.get(key)):
                raise ValueError(f"Invalid hash: {key}")
        expected_name = f"state8-label-readout-{tag}"
        if job.get("name") != expected_name:
            raise ValueError("Job name drift")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, expected_name)
        if runner_sha not in command or f"bash scripts/{RUNNER}" not in command:
            raise ValueError("Runner binding drift")
    if observed != set(SPECS):
        raise ValueError("Readout matrix drift")
    if any(fragment in rendered for fragment in ("optimizer.step()", "--init-adapter")):
        raise ValueError("Read-only probe mutates weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
