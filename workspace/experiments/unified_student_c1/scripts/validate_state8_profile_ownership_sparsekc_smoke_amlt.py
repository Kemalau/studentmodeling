#!/usr/bin/env python3
"""Fail-closed validator for the bounded sparse-KC ownership smoke."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from validate_state8_curriculum_checkpoint_screen_amlt import (
        IMAGE,
        WORKSPACE,
        validate_code_root,
    )
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )
    from scripts.validate_state8_curriculum_checkpoint_screen_amlt import (
        IMAGE,
        WORKSPACE,
        validate_code_root,
    )


ROOT = Path(__file__).resolve().parents[1]
RUNNER = "run_state8_student_profile_readout.sh"
JOB = "state8-profile-ownership-lrfix-v4-sparsekc-smoke-r2-seed8801"
STORAGE = {
    "storage_account_name": "csnewsandfeeds4150361735",
    "container_name": "amulet",
    "mount_dir": "/mnt/amulet",
    "is_output": True,
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b|PENDING|RUNNER_SHA256", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = config.get("code") or {}
    if code.get("local_dir") != "$CONFIG_DIR/..":
        validate_code_root(config)
    if (config.get("storage") or {}).get("student_store") != STORAGE:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job cardinality/name drift")
    job = jobs[0]
    if (
        job.get("sku"), job.get("process_count_per_node"),
        job.get("mpi"), job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
        raise ValueError("Retry/duration drift")
    if (submit.get("env") or {}) != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-profile-ownership-lrfix-venv",
    }:
        raise ValueError("Identity/environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    runner_hash = hashlib.sha256((ROOT / "scripts" / RUNNER).read_bytes()).hexdigest()
    required = (
        runner_hash,
        f"bash scripts/{RUNNER}",
        "PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS=200",
        "STATE8_PROFILE_OWNERSHIP_HEAD_LEARNING_RATE=3e-4",
        "STATE8_PROFILE_OWNERSHIP_RESET_HEAD=1",
        "STATE8_PROFILE_OWNERSHIP_FEATURE_NORMALIZATION=layernorm",
        "STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT=4",
        "STATE8_PROFILE_OWNERSHIP_BIAS_PRIOR_STRENGTH=8",
        "state8_profile_ownership_lrfix_v4_sparsekc_smoke_r2_seed8801",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Sparse-KC smoke command binding drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
