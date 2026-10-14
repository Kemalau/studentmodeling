#!/usr/bin/env python3
"""Fail-closed validation for the query-aligned State8 Eedi curriculum."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-diagnosis-first-eedi-current-query-curriculum-seed83"


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
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku"), job.get("process_count_per_node"),
        job.get("mpi"), job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
        raise ValueError("Retry/duration drift")
    env = submit.get("env") or {}
    if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if "run_state8_diagnosis_first_eedi_current_query_curriculum.sh" not in command:
        raise ValueError("Entrypoint drift")
    scripts = Path(__file__).resolve().parent
    source = "\n".join(
        (scripts / name).read_text(encoding="utf-8")
        for name in (
            "run_state8_diagnosis_first_eedi_current_query_curriculum.sh",
            "train_state8_diagnosis_first_eedi_current_query_curriculum.py",
            "state8_current_query_routing.py",
        )
    )
    for fragment in (
        "current_item_visible_response_state_only_v1:eedi_only",
        "student_response_visible_to_decoder",
        "raw_history_visible_to_decoder",
        "history_dropout_schedule",
        "same_question_causal_replay_required",
        "semantic_judge_authoritative",
        "exact_match_authoritative",
    ):
        if fragment not in source:
            raise ValueError(f"Current-query contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
