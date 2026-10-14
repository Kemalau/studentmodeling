#!/usr/bin/env python3
"""Fail-closed validator for the matched Eedi-dense objective diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
MANIFEST = "manifests/layerwise_task_retention_eedi_dense_seed42_v1.json"
MANIFEST_SHA256 = "cc484d7f7b6db02cfdf1e530866e029a721896d9c57103313e581f504c7bc16e"
JOBS = {
    "layerwise-eedi-dense-standard-seed42": "standard",
    "layerwise-eedi-dense-state-path-seed42": "state_path",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
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

    expected = set(JOBS) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(JOBS):
        raise ValueError("Expected job-set drift")
    jobs = config.get("jobs") or []
    if len(jobs) != len(expected) or {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("Job-set drift")
    commands = []
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G4-A100"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("VENV_ROOT") != "/tmp/studentmodeling-layerwise-eedi-ab-venv":
            raise ValueError(f"Venv drift: {name}")
        observed = (
            str(env.get("BRANCH")),
            str(env.get("LOSS_PROFILE")),
            str(env.get("MANIFEST")),
            str(env.get("MANIFEST_SHA256")),
        )
        expected_arm = ("eedi_dense", JOBS[name], MANIFEST, MANIFEST_SHA256)
        if observed != expected_arm:
            raise ValueError(f"Controlled arm drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Matched jobs do not share one command")
    if commands[0].strip() != "set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh":
        raise ValueError("Launcher drift")

    root = Path(__file__).resolve().parent
    launcher = (root / "run_layerwise_task_retention_rehearsal.sh").read_text(encoding="utf-8")
    required = (
        'VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-layerwise-task-retention-venv}"',
        'python() { "$VENV_ROOT/bin/python" "$@"; }',
        "import accelerate,bitsandbytes,peft,safetensors,transformers",
        'LOSS_PROFILE="${LOSS_PROFILE:-standard}"',
        "BEHAVIOR_KD_WEIGHT=0.0",
        "DIRECT_REPLAY_WEIGHT=0.0",
        '--generation-weight 1.0 --consistency-weight 0.25',
        "--causal-objective-mode specificity_triplet",
        "--diagnostic-warm-start-checkpoint \"$CHECKPOINT/recurrent_state.pt\"",
        "--diagnostic-index-manifest \"$MANIFEST\"",
        '--learning-rate "$LEARNING_RATE" --lr-scheduler constant',
        '--diagnostic-stop-after-round "$DIAGNOSTIC_STOP_AFTER_ROUND"',
        "--tasks predict_error_persistence",
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440",
        "20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a",
    )
    if any(fragment not in launcher for fragment in required):
        raise ValueError("Launcher scientific invariant drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
