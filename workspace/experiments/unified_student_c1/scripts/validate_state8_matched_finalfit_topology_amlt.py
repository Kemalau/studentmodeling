#!/usr/bin/env python3
"""Fail-closed validation for compact matched-State8 final-fit topologies."""

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
JOB = "state8-diagnosis-first-matched-seed67-unified-finalfit-seed42"
TOPOLOGIES = {
    "80G4-A100": "run_state8_unified_final_fit_world4.sh",
    "80G2-A100": "run_state8_unified_final_fit_world2.sh",
}


def validate(config: dict) -> None:
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
    code = config.get("code") or {}
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Matched job set drift")
    job = jobs[0]
    launcher = TOPOLOGIES.get(str(job.get("sku")))
    if launcher is None:
        raise ValueError("Topology drift")
    if (
        int(job.get("process_count_per_node", 0)) != -1
        or job.get("mpi") is not True
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Distributed topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
        raise ValueError("Retry/duration drift")
    if submit.get("env") != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-finalfit-venv",
        "FINALFIT_VARIANT": "diagnosis_first_matched_seed67",
    }:
        raise ValueError("Environment/UAI/variant drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    for fragment in (
        "set -euo pipefail",
        "FAILED line=%s rc=%s command=%s",
        f"bash scripts/{launcher}",
    ):
        if fragment not in command:
            raise ValueError(f"Launcher drift: {fragment}")
    root = Path(__file__).resolve().parent
    base = (root / "run_state8_unified_final_fit.sh").read_text(encoding="utf-8")
    for fragment in (
        "diagnosis_first_matched_seed67",
        "state8_diagnosis_first_matched_joint_promotion_v1",
        "state8_diagnosis_first_matched_unified_full259845_seed42_v1_20260810",
        "state8_diagnosis_first_matched_unified_canonical_v1_20260810",
        "history_dropout_counts",
        "completion['micro_steps']==259845",
        "--causal-objective-mode specificity_triplet",
    ):
        if fragment not in base:
            raise ValueError(f"Matched final-fit invariant drift: {fragment}")
    wrapper = (root / launcher).read_text(encoding="utf-8")
    expected_renderer = {
        "run_state8_unified_final_fit_world4.sh": "render_state8_finalfit_world4_launcher.py",
        "run_state8_unified_final_fit_world2.sh": "render_state8_finalfit_world2_launcher.py",
    }[launcher]
    if expected_renderer not in wrapper:
        raise ValueError("Compact topology wrapper drift")


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
