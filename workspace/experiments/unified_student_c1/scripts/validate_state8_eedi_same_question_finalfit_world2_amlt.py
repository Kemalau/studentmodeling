#!/usr/bin/env python3
"""Fail-closed validation for the promoted same-question world2 final fit."""

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
VARIANTS = {
    "same_question_updater_mse1_seed3072",
    "same_question_full_mse1_seed3071",
    "same_question_repair_promoted",
    "same_question_repair_checkpoint_promoted",
    "same_question_curriculum_promoted",
}


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"SELECTED_VARIANT|<placeholder>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
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
    if len(jobs) != 1:
        raise ValueError("Expected one selected final-fit job")
    job = jobs[0]
    if (
        job.get("sku") != "80G2-A100"
        or int(job.get("process_count_per_node", 0)) != -1
        or job.get("mpi") is not True
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 345600:
        raise ValueError("Retry/duration drift")
    env = submit.get("env") or {}
    if set(env) != {"_AZUREML_SINGULARITY_JOB_UAI", "VENV_ROOT", "FINALFIT_VARIANT"}:
        raise ValueError("Environment schema drift")
    if env["_AZUREML_SINGULARITY_JOB_UAI"] != "rankfun_aml":
        raise ValueError("UAI drift")
    if env["FINALFIT_VARIANT"] not in VARIANTS:
        raise ValueError("Variant drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, str(job.get("name")))
    for fragment in (
        "set -euo pipefail",
        "render_state8_eedi_same_question_finalfit_world2.py",
        "736d06fd7596707c9a669f9bff0a649d34134c40814fc5c88f77b268e61d1615",
        "8d025cb0301b894b85c1e2eac34bc3f318dc269fc76bd581b8aaa2bc887cc59f",
        "exec bash",
    ):
        if fragment not in command:
            raise ValueError(f"Launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    value = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    validate(value)
    print("valid=true")


if __name__ == "__main__":
    main()
