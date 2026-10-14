#!/usr/bin/env python3
"""Fail-closed validator for the lrfix-v4 full DBE development run."""

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
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from scripts.validate_state8_curriculum_checkpoint_screen_amlt import (
        IMAGE,
        WORKSPACE,
        validate_code_root,
    )


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_profile_ownership_lrfix_v4_formal_world4_20260821.yaml"
RUNNER = ROOT / "scripts/run_state8_student_profile_readout.sh"
JOB = "state8-profile-ownership-lrfix-v4-formal-world4-seed8801"
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
    if code.get("local_dir") == "$CONFIG_DIR/..":
        pass
    else:
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
        job.get("sku"), job.get("process_count_per_node"),
        job.get("mpi"), job.get("sla_tier"),
    ) != ("80G4-A100", -1, True, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    if (submit.get("env") or {}) != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-profile-ownership-lrfix-v4-formal-world4-venv",
    }:
        raise ValueError("Identity/environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    if digest not in command or "bash scripts/run_state8_student_profile_readout.sh" not in command:
        raise ValueError("Runner binding drift")
    if "PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS=" in command:
        raise ValueError("Formal run is accidentally bounded as a diagnostic")
    if "unset PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS" not in command:
        raise ValueError("Formal diagnostic environment is not cleared")
    if "manifests/xes_state8_profile_confirmatory_v1/test.jsonl" in command:
        raise ValueError("Frozen XES confirmatory test leaked into development")
    if "export PROFILE_READOUT_DISTRIBUTED_WORLD_SIZE=4" not in command:
        raise ValueError("Formal world4 run lacks the four-rank contract")
    if "export PROFILE_READOUT_GRADIENT_ACCUMULATION_STEPS=1" not in command:
        raise ValueError("Formal world4 run no longer preserves global batch four")
    for name, expected in EXPECTED_ENV.items():
        if f"export {name}={expected}" not in command:
            raise ValueError(f"Repair setting drift: {name}")
    if "/state8_profile_ownership_lrfix_v4_formal_world4_seed8801" not in command:
        raise ValueError("Formal remote output drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
