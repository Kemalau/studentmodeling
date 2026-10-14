#!/usr/bin/env python3
"""Fail-closed validator for exact diagnosis-first milestone evaluations."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
STEPS = (200, 400, 600, 800)


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = config.get("code") or {}
    if code.get("local_dir") != "$CONFIG_DIR/..":
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    expected_names = {f"state8-exact-consolidation-step{step}-eval" for step in STEPS}
    if {str(job.get("name")) for job in jobs} != expected_names:
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        step = int(name.split("step", 1)[1].split("-", 1)[0])
        if (
            job.get("sku"), job.get("process_count_per_node"),
            job.get("mpi"), job.get("sla_tier")
        ) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        expected_env = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-exact-milestone-venv",
            "MILESTONE_STEP": str(step),
        }
        if {key: str(value) for key, value in env.items()} != expected_env:
            raise ValueError(f"Environment drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
            raise ValueError(f"Retry policy drift: {name}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if "run_state8_exact_consolidation_milestone_eval.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    launcher = Path(__file__).with_name(
        "run_state8_exact_consolidation_milestone_eval.sh"
    ).read_text(encoding="utf-8")
    for fragment in (
        "case \"$MILESTONE_STEP\" in 200|400|600|800)",
        "planned_optimizer_steps']==250",
        "history_dropout_schedule']==[1.0]",
        "diagnosis_field_order']==['diagnosis','evidence','label']",
        "evaluate_recurrent_formal_protocols_diagnosis_first.py",
        "evaluate_recurrent_causal_gate.py",
        "evaluate_recurrent_diagnosis_c4_diagnosis_first.py",
        "early_stop_candidate_only",
        "raw_history_visible_to_decoder",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
