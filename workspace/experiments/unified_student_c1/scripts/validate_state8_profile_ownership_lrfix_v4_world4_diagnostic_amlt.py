#!/usr/bin/env python3
"""Fail-closed validator for the exact lrfix-v4 four-rank preflight."""

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
except ImportError:
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
CONFIG = ROOT / "amlt/feeds_state8_profile_ownership_lrfix_v4_world4_diagnostic_20260821.yaml"
RUNNER = ROOT / "scripts/run_state8_student_profile_readout.sh"
JOB = "state8-profile-ownership-lrfix-v4-world4-diagnostic-seed8801"
EXPECTED_ENV = {
    "STATE8_PROFILE_OWNERSHIP_HEAD_LEARNING_RATE": "3e-4",
    "STATE8_PROFILE_OWNERSHIP_RESET_HEAD": "1",
    "STATE8_PROFILE_OWNERSHIP_FEATURE_NORMALIZATION": "layernorm",
    "STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT": "4",
    "STATE8_PROFILE_OWNERSHIP_BIAS_PRIOR_STRENGTH": "8",
}


def validate(path: Path = CONFIG) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(value, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b|PENDING|RUNNER_SHA256", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(value.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (value.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = value.get("code") or {}
    if code.get("local_dir") != "$CONFIG_DIR/..":
        validate_code_root(value)
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
    if (
        job.get("sku"),
        job.get("process_count_per_node"),
        job.get("mpi"),
        job.get("sla_tier"),
    ) != ("80G4-A100", -1, True, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 1 or submit.get("max_run_duration_seconds") != 14400:
        raise ValueError("Retry/duration drift")
    if (submit.get("env") or {}) != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-profile-ownership-lrfix-v4-world4-diagnostic-venv",
    }:
        raise ValueError("Identity/environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    required = (
        digest,
        "bash scripts/run_state8_student_profile_readout.sh",
        "PROFILE_READOUT_DISTRIBUTED_WORLD_SIZE=4",
        "PROFILE_READOUT_GRADIENT_ACCUMULATION_STEPS=1",
        "PROFILE_READOUT_DIAGNOSTIC_STOP_ROUND=2",
        "state8_profile_ownership_lrfix_v4_world4_diagnostic_seed8801",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("World4 diagnostic command binding drift")
    for name, expected in EXPECTED_ENV.items():
        if f"export {name}={expected}" not in command:
            raise ValueError(f"Repair setting drift: {name}")
    if "manifests/xes_state8_profile_confirmatory_v1/test.jsonl" in command:
        raise ValueError("Frozen XES confirmatory test leaked into execution preflight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
