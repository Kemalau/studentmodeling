#!/usr/bin/env python3
"""Fail-closed AMLT validation for the State8 label-first repair."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from validate_state8_curriculum_checkpoint_screen_amlt import (
    IMAGE,
    WORKSPACE,
    validate_code_root,
)


RUNNER = "run_state8_label_first_repair.sh"
JOB = "state8-label-first-repair-seed8501"


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
    validate_code_root(config)
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Exactly one label-first repair job is required")
    job = jobs[0]
    if job.get("name") != JOB:
        raise ValueError("Job name drift")
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    env = submit.get("env") or {}
    fixed = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "SOURCE_ROOT": "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed8401_taskanchor_source_anchor_full_mse1_eedi_same_question_v1_early_label_high_seed8401",
        "EXPECTED_STUDENT_ADAPTER_SHA256": "3bcf7bdc1d6e4de432efbd2bf78551c3d92349e44d7e2b217166ca1305cffe24",
        "EXPECTED_STUDENT_STATE_SHA256": "6985e40ec82e100211a552847b2f4a8601fbdb5785d58fdcb0f0620df10d5762",
        "REPAIR_REMOTE_ROOT": "/mnt/amulet/studentmodeling/diagnostics/state8_label_first_repair_seed8501_v1",
        "STATE8_REPAIR_MANIFEST": "manifests/state8_label_first_repair_seed8501_v1.json",
        "STATE8_REPAIR_MANIFEST_SHA256": "4a5e31add8732a16ccc109ad9ff3d1956e9c0dcfa4608a0dea3800d7ee295f9c",
    }
    if any(str(env.get(key)) != value for key, value in fixed.items()):
        raise ValueError("Environment binding drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    runner_sha = hashlib.sha256(
        (Path(__file__).resolve().parent / RUNNER).read_bytes()
    ).hexdigest()
    if runner_sha not in command or f"bash scripts/{RUNNER}" not in command:
        raise ValueError("Runner binding drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
