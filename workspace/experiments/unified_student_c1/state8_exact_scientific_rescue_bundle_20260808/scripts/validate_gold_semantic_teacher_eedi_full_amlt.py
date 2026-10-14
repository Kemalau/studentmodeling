#!/usr/bin/env python3
"""Fail-closed preflight for the gold-semantic state-prototype diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
TEACHER = "/mnt/amulet/studentmodeling/adapters/qwen3_32b_full_trainvalid_no_t4_strictt6_1ep/final_adapter"
TEACHER_SHA256 = "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a"
ENTRYPOINT_SHA256 = "62f692c96159b55f2393ee33e6c759fc5bc64363a5a42034eb94d886a79a9541"
JOBS = {
    "gold-semantic-teacher-cw025-seed42": "0.25",
    "gold-semantic-teacher-cw100-seed42": "1.0",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
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
    storage = (config.get("storage") or {}).get("student_store")
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
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code upload excludes drift")

    expected = set(JOBS) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(JOBS):
        raise ValueError("Expected jobs drift")
    jobs = config.get("jobs") or []
    if len(jobs) != len(expected) or {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G4-A100"
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        required = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-gold-semantic-teacher-venv",
            "BRANCH": "eedi_full_open_t3",
            "LOSS_PROFILE": "state_path",
            "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_gold_semantic_teacher.py",
            "SEGMENT_ROWS": "5000",
            "LEARNING_RATE": "2e-6",
            "CONSISTENCY_WEIGHT": JOBS[name],
            "EXPECTED_CAUSAL_PAIRS": "250",
            "DIAGNOSTIC_STOP_AFTER_ROUND": "4250",
            "MANIFEST": "/tmp/layerwise_extended_manifests/layerwise_task_retention_eedi_full_seed42_v1.json",
            "MANIFEST_SHA256": "97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393",
            "STUDENTMODELING_EXTERNAL_STATE_TEACHER_ADAPTER": TEACHER,
            "STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256": TEACHER_SHA256,
        }
        if {key: str(env.get(key)) for key in required} != required:
            raise ValueError(f"Controlled arm drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if command.strip() != "set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh":
            raise ValueError("Launcher drift")

    root = Path(__file__).resolve().parent
    entrypoint = root / "train_recurrent_student_m4_gold_semantic_teacher.py"
    if hashlib.sha256(entrypoint.read_bytes()).hexdigest() != ENTRYPOINT_SHA256:
        raise ValueError("Gold-semantic entrypoint hash drift")
    launcher = (root / "run_layerwise_task_retention_rehearsal.sh").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "scripts/train_recurrent_student_m4_gold_semantic_teacher.py",
        "EXPECTED_ENTRYPOINT_SHA256=62f692c96159b55f2393ee33e6c759fc5bc64363a5a42034eb94d886a79a9541",
        "gold_semantic_eedi_t3_layerwise_trajectory",
        "gold_semantic_state_teacher_training_view.json",
        "--causal-objective-mode specificity_triplet",
    ):
        if fragment not in launcher:
            raise ValueError(f"Gold-semantic launcher invariant drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
