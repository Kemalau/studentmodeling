#!/usr/bin/env python3
"""Fail-closed validation for the selected same-question answer-C4 job."""

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


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"MATERIALIZATION_REQUIRED|<placeholder>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store")
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != "state8-eedi-same-question-answer-c4-selected":
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
    arm = str(env.get("SAME_QUESTION_ARM") or "")
    layout = str(env.get("SAME_QUESTION_ARTIFACT_LAYOUT") or "")
    old_arm = arm in {"updater", "full"} or arm.startswith("repair_")
    curriculum_arm = arm.startswith("curriculum_")
    if not (old_arm or curriculum_arm):
        raise ValueError("Arm drift")
    if layout not in {
        "repair_publish", "curriculum_snapshot", "repair_checkpoint_snapshot"
    }:
        raise ValueError("Artifact layout drift")
    repair_checkpoint = layout == "repair_checkpoint_snapshot"
    if curriculum_arm != (layout == "curriculum_snapshot"):
        raise ValueError("Arm/layout mismatch")
    if repair_checkpoint != arm.startswith("repair_checkpoint_"):
        raise ValueError("Repair checkpoint arm/layout mismatch")
    root = str(env.get("SAME_QUESTION_REMOTE_ROOT") or "")
    valid_root = (
        root.startswith(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
        )
        if repair_checkpoint
        else (
        root.startswith(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_diagnosis_first_eedi_"
        )
        if curriculum_arm
        else root.startswith(
            "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed"
        )
        )
    )
    if not valid_root:
        raise ValueError("Remote root drift")
    for key in ("SAME_QUESTION_ADAPTER_SHA256", "SAME_QUESTION_STATE_SHA256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(env.get(key) or "")):
            raise ValueError(f"Invalid hash: {key}")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, job["name"])
    if "bash scripts/run_state8_eedi_same_question_answer_c4.sh" not in command:
        raise ValueError("Runner drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
