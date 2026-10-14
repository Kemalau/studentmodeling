#!/usr/bin/env python3
"""Fail-closed validation for the State8 Eedi curriculum AMLT job."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-diagnosis-first-eedi-curriculum-seed83"
UAI = "rankfun_aml"


def validate(path: Path) -> None:
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
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Remote storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku"),
        job.get("process_count_per_node"),
        job.get("mpi"),
        job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
        raise ValueError("Retry/duration drift")
    if (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
        raise ValueError("UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if "run_state8_diagnosis_first_eedi_curriculum.sh" not in command:
        raise ValueError("Entrypoint drift")
    source = (
        Path(__file__).resolve().parent
        / "run_state8_diagnosis_first_eedi_curriculum.sh"
    ).read_text(encoding="utf-8")
    parent = (
        Path(__file__).resolve().parent
        / "run_state8_qualified_teacher_curriculum.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "state_token_count",
        "STATE8_CAUSAL_REPEAT_FRACTION=0.10",
        "train_state8_diagnosis_first_eedi_curriculum.py",
        "eedi_same_question_c4_25",
        "same_question_query_identical",
        "raw_history_visible_to_decoder",
        "semantic_judgment_pending",
        "rankfun_aml",
    ):
        haystack = source + parent + rendered
        if fragment not in haystack:
            raise ValueError(f"Eedi curriculum contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
