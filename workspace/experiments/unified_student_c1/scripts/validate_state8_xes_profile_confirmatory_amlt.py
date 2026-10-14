#!/usr/bin/env python3
"""Fail-closed validation for the one-shot frozen XES C4 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from validate_state8_curriculum_checkpoint_screen_amlt import IMAGE, WORKSPACE
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from scripts.validate_state8_curriculum_checkpoint_screen_amlt import IMAGE, WORKSPACE


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_xes_profile_confirmatory_20260820.yaml"
RUNNER = ROOT / "scripts/run_state8_xes_profile_confirmatory.sh"
JOB = "state8-xes-profile-confirmatory-lrfix-v4-seed8801"


def validate(path: Path = CONFIG) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(value, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b|PENDING|RUNNER_SHA256", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(value.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (value.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (value.get("storage") or {}).get("student_store")
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = value.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job cardinality/name drift")
    job = jobs[0]
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 1 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("One-shot retry/duration drift")
    if (submit.get("env") or {}) != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-xes-profile-confirmatory-venv",
    }:
        raise ValueError("Identity/environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if hashlib.sha256(RUNNER.read_bytes()).hexdigest() not in command:
        raise ValueError("Runner hash drift")
    source = RUNNER.read_text(encoding="utf-8")
    required = (
        "verify_xes_profile_confirmatory_freeze.py",
        "state8_profile_ownership_lrfix_v4_formal_world4_seed8801",
        "manifests/xes_state8_profile_confirmatory_v1",
        "--bootstrap-replicates 5000",
        "--permutation-replicates 2000",
        "test_labels_used_for_selection",
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("Confirmatory protocol binding drift")
    forbidden = ("optimizer.step()", "--init-adapter", "train_state8_", "amlt run")
    if any(fragment in source for fragment in forbidden):
        raise ValueError("Confirmatory evaluator mutates or resubmits")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
