#!/usr/bin/env python3
"""Fail-closed validation for the two-A100 State8 final-fit fallback."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "state8-strong1-widen8-lr5e7-seed58-unified-finalfit-seed42":
        "strong1_widen8_lr5e7_seed58",
    "state8-strong1-widen8-lr1e6-seed59-unified-finalfit-seed42":
        "strong1_widen8_lr1e6_seed59",
    "state8-strong1-widen8-single-lr5e7-seed60-unified-finalfit-seed42":
        "strong1_widen8_single_lr5e7_seed60",
    "state8-strong1-rescue-taxonomy-joint-seed61-unified-finalfit-seed42":
        "strong1_rescue_taxonomy_joint_seed61",
    "state8-strong1-rescue-taskanchor-c4strong-seed62-unified-finalfit-seed42":
        "strong1_rescue_taskanchor_c4strong_seed62",
    "state8-strong1-midpoint-diagnosis-first-seed63-unified-finalfit-seed42":
        "strong1_midpoint_diagnosis_first_seed63",
    "state8-strong1-exact-diagnosis-curriculum-seed65-unified-finalfit-seed42":
        "strong1_exact_diagnosis_curriculum_seed65",
}


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
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
    if not jobs or len({job.get("name") for job in jobs}) != len(jobs):
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job.get("name"))
        variant = JOBS.get(name)
        if variant is None:
            raise ValueError(f"Unexpected variant: {name}")
        if (
            job.get("sku") != "80G2-A100"
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
            raise ValueError("Retry/duration drift")
        env = submit.get("env") or {}
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-finalfit-venv",
            "FINALFIT_VARIANT": variant,
            "STATE8_CANONICAL_RUN_DIR": (
                "/mnt/amulet/studentmodeling/checkpoints/"
                "state8_strong1_widen8_unified_full259845_seed42_v1_20260808"
            ),
        }
        if env != expected:
            raise ValueError("Environment/UAI/variant drift")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        for fragment in (
            "set -euo pipefail",
            "FAILED line=%s rc=%s command=%s",
            "bash scripts/run_state8_unified_final_fit_world2.sh",
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
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
