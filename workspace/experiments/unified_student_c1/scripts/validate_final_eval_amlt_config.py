#!/usr/bin/env python3
"""Fail-closed validation for final-fit T1/T2/T3/T5/T6 and C4 evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
EXPECTED = {
    "final-eval-t1-t6-seed42": ("t1_t6", "predict_correctness predict_error_persistence"),
    "final-eval-t2-seed42": ("t2", "predict_response"),
    "final-eval-t3-seed42": ("t3", "diagnose_misconception"),
    "final-eval-t5-seed42": ("t5", "predict_prospective_error"),
    "final-eval-c4-seed42": (None, None),
}


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [index for marker in ("description:", "target:") if (index := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|PLACEHOLDER", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {"storage_account_name": "csnewsandfeeds4150361735", "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True}:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if {job.get("name") for job in jobs} != set(EXPECTED) or len(jobs) != len(EXPECTED):
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        if job.get("sku") != "80G1-A100" or job.get("mpi") is not False or int(job.get("process_count_per_node", 0)) != 1:
            raise ValueError(f"GPU/MPI drift in {name}")
        if job.get("sla_tier") != "Standard":
            raise ValueError(f"SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or env.get("_AZUREML_SINGULARITY_JOB_UAI") != UAI:
            raise ValueError(f"Retry/UAI drift in {name}")
        group, tasks = EXPECTED[name]
        if group is not None and (env.get("EVAL_GROUP") != group or env.get("TASKS") != tasks):
            raise ValueError(f"Task label drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        common = (
            "final_fit_complete",
            "real_examples') == 259845",
            "causal_replay_fraction') == 0.05",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "9e0431dccb524f67b401f7ea911e11c96fc3061432ffdc73e0cac2bbc2acc5a0",
        )
        specific = (
            "evaluate_recurrent_causal_gate.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
        ) if group is None else (
            "evaluate_recurrent_formal_protocols.py",
            "eb0243cf232d7e634f0ad784d30e66f93f42f84833a748ec453b648cff0eccda",
            "a368cefb0c49cb6a305ad1a892808361fba2d44d555a0a28ea64e56babe32525",
            "--save-unredacted-predictions",
        )
        if any(value not in command for value in (*common, *specific)):
            raise ValueError(f"Command drift in {name}")
        forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models", "msrcommonacr.azurecr.io/ml-common-py3.12:latest")
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Secret/local/forbidden path drift in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load_config(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
