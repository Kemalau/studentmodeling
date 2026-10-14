#!/usr/bin/env python3
"""Fail-closed AMLT validator for the Teacher checkpoint-200 semantic screen."""

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
JOB = "state8-diagnosis-first-teacher-checkpoint200-screen"


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
    store = (config.get("storage") or {}).get("student_store")
    if store != {"storage_account_name":"csnewsandfeeds4150361735","container_name":"amulet","mount_dir":"/mnt/amulet","is_output":True}:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]; submit = job.get("submit_args") or {}; env = submit.get("env") or {}
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != ("80G1-A100",1,False,"Standard"):
        raise ValueError("Topology drift")
    if env != {"_AZUREML_SINGULARITY_JOB_UAI":"rankfun_aml"}:
        raise ValueError("UAI drift")
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
        raise ValueError("Retry/duration drift")
    command = command_text(job.get("command")); validate_shell_syntax(command, JOB)
    for fragment in (
        "checkpoint-200", "trainer_state.json", "x['global_step']==200",
        "build_recurrent_evaluation_manifest.py", "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
        "build_state8_diagnosis_first_teacher_views import write_view",
        "add_state8_teacher_markers.py", "--max-samples-per-group 25",
        "--save-unredacted-predictions", "ADAPTER_SHA256",
    ):
        if fragment not in command:
            raise ValueError(f"Protocol drift: {fragment}")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True)
    args=parser.parse_args(); validate(args.config); print(f"validated={args.config}")


if __name__ == "__main__":
    main()
