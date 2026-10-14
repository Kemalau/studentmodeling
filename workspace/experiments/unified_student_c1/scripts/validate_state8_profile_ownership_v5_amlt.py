#!/usr/bin/env python3
"""Fail-closed validation for State8 ownership-v5 DBE runs."""

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
RUNNER = ROOT / "scripts/run_state8_student_profile_readout_v5.sh"
STORAGE = {
    "storage_account_name": "csnewsandfeeds4150361735",
    "container_name": "amulet",
    "mount_dir": "/mnt/amulet",
    "is_output": True,
}
CONTRACT = (
    "STATE8_PROFILE_OWNERSHIP_HEAD_LEARNING_RATE=3e-4",
    "STATE8_PROFILE_OWNERSHIP_RESET_HEAD=1",
    "STATE8_PROFILE_OWNERSHIP_FEATURE_NORMALIZATION=layernorm",
    "STATE8_PROFILE_OWNERSHIP_MIN_KC_TRAIN_SUPPORT=4",
    "STATE8_PROFILE_OWNERSHIP_BIAS_PRIOR_STRENGTH=8",
    "STATE8_V5_PROFILE_GENERATION_WEIGHT=0.1",
    "STATE8_PROFILE_LEGACY_CAUSAL_ENABLED=0",
    "bash scripts/run_state8_student_profile_readout_v5.sh",
)


def validate(path: Path, mode: str) -> None:
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
    # AMLT 11.14.2 resolves local_dir into storage_id in --dump output.
    if not (
        code.get("local_dir") == "$CONFIG_DIR/.."
        or (code.get("storage_id") == "default" and "local_dir" not in code)
    ):
        raise ValueError("Code-root drift")
    if (value.get("storage") or {}).get("student_store") != STORAGE:
        raise ValueError("Storage drift")
    jobs = value.get("jobs") or []
    expected_job = f"state8-profile-ownership-v5-{mode}-seed8801"
    if len(jobs) != 1 or jobs[0].get("name") != expected_job:
        raise ValueError("Job cardinality/name drift")
    job = jobs[0]
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    expected_duration = 43200 if mode == "smoke" else 172800
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != expected_duration:
        raise ValueError("Retry/duration drift")
    expected_venv = f"/tmp/studentmodeling-state8-profile-ownership-v5-{mode}-venv"
    if (submit.get("env") or {}) != {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": expected_venv,
    }:
        raise ValueError("Identity/environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, expected_job)
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    if digest not in command or any(fragment not in command for fragment in CONTRACT):
        raise ValueError("V5 command binding drift")
    if "manifests/xes_state8_profile_confirmatory_v1/test.jsonl" in command:
        raise ValueError("Frozen XES test leaked into DBE development")
    if mode == "smoke":
        if "PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS=200" not in command:
            raise ValueError("Smoke is not bounded to 200 micro-steps")
    else:
        if "unset PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS" not in command:
            raise ValueError("Formal run does not clear diagnostic state")
        if "PROFILE_READOUT_DIAGNOSTIC_MAX_MICRO_STEPS=200" in command:
            raise ValueError("Formal run is accidentally bounded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    validate(args.config, args.mode)
    print(f"validated={args.config} mode={args.mode}")


if __name__ == "__main__":
    main()
