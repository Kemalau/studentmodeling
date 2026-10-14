#!/usr/bin/env python3
"""Fail-closed validator for equal T3/T6 + open-T3 state-path training."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-equal-t3-t6-open-t3-state-path-seed42"
MANIFEST = "manifests/layerwise_task_retention_equal_t3_t6_seed42_v1.json"
MANIFEST_SHA256 = "32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


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
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
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
        raise ValueError("Code-upload root drift")
    if not {"data/", "results/", ".git/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code upload can leak local artifacts")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job-set drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G4-A100"
        or job.get("mpi") is not True
        or int(job.get("process_count_per_node", 0)) != -1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    expected_env = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-layerwise-equal-open-statepath-venv",
        "BRANCH": "equal_t3_t6_open_t3",
        "LOSS_PROFILE": "state_path",
        "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_open_t3.py",
        "MANIFEST": MANIFEST,
        "MANIFEST_SHA256": MANIFEST_SHA256,
    }
    if any(str(env.get(key)) != value for key, value in expected_env.items()):
        raise ValueError("Controlled-arm environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if command.strip() != "set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh":
        raise ValueError("Launcher drift")
    scripts = Path(__file__).resolve().parent
    launcher = (scripts / "run_layerwise_task_retention_rehearsal.sh").read_text(encoding="utf-8")
    required = (
        "equal_t3_t6_open_t3",
        "scripts/train_recurrent_student_m4_open_t3.py",
        "BEHAVIOR_KD_WEIGHT=0.0",
        "DIRECT_REPLAY_WEIGHT=0.0",
        '--diagnostic-index-manifest "$MANIFEST"',
        "--state-architecture layerwise",
        "--causal-objective-mode specificity_triplet",
        "--save-steps 50 --checkpoint-retention 2",
        "--tasks predict_error_persistence",
    )
    if any(fragment not in launcher for fragment in required):
        raise ValueError("Launcher scientific invariant drift")
    wrapper = (scripts / "train_recurrent_student_m4_open_t3.py").read_text(encoding="utf-8")
    wrapper_required = (
        "apply_open_t3_training_view",
        'if cli_float("--behavior-kd-weight", 0.0) != 0.0',
        'if cli_float("--direct-replay-weight", 0.0) != 0.0',
        '"causal_rows_modified": False',
        '"validation_rows_modified": False',
    )
    if any(fragment not in wrapper for fragment in wrapper_required):
        raise ValueError("Open-T3 wrapper drift")
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models")
    if any(fragment.lower() in launcher.lower() for fragment in forbidden):
        raise ValueError("Secret/local model path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
