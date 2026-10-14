#!/usr/bin/env python3
"""Fail-closed validator for the closed-state KD diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-eedi-dense-state-kd-seed42"
MANIFEST = "manifests/layerwise_task_retention_eedi_dense_seed42_v1.json"
MANIFEST_SHA256 = "cc484d7f7b6db02cfdf1e530866e029a721896d9c57103313e581f504c7bc16e"


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
        raise ValueError("Job drift")
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
        "VENV_ROOT": "/tmp/studentmodeling-layerwise-state-kd-venv",
        "BRANCH": "eedi_dense",
        "LOSS_PROFILE": "state_kd",
        "MANIFEST": MANIFEST,
        "MANIFEST_SHA256": MANIFEST_SHA256,
    }
    if any(str(env.get(key)) != value for key, value in expected_env.items()):
        raise ValueError("Controlled arm drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if command.strip() != "set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh":
        raise ValueError("Launcher drift")
    launcher = (Path(__file__).resolve().parent / "run_layerwise_task_retention_rehearsal.sh").read_text(encoding="utf-8")
    required = (
        "state_kd)",
        "BEHAVIOR_KD_WEIGHT=0.5",
        "DIRECT_REPLAY_WEIGHT=0.0",
        "--causal-objective-mode specificity_triplet",
        '--diagnostic-index-manifest "$MANIFEST"',
        "'state_kd':(0.5,0.0)",
    )
    if any(fragment not in launcher for fragment in required):
        raise ValueError("Launcher state-KD invariant drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
