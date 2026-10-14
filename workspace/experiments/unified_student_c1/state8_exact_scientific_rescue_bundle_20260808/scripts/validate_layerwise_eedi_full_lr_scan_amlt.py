#!/usr/bin/env python3
"""Fail-closed AMLT validator for the full-support Eedi LR scan."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
MANIFEST = "/tmp/layerwise_extended_manifests/layerwise_task_retention_eedi_full_seed42_v1.json"
MANIFEST_SHA256 = "97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393"
JOBS = {
    "layerwise-eedi-full-statepath-lr2e6-seed42": "2e-6",
    "layerwise-eedi-full-statepath-lr5e6-seed42": "5e-6",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


def validate(
    config: dict,
    expected_jobs: set[str] | None = None,
    *,
    jobs_by_lr: dict[str, str] | None = None,
    loss_profile: str = "state_path",
    venv_root: str = "/tmp/studentmodeling-layerwise-eedi-full-lr-venv",
    branch: str = "eedi_full",
    train_entrypoint: str | None = None,
    segment_rows: str = "5000",
    expected_causal_pairs: str = "250",
    diagnostic_stop_after_round: str = "4250",
    manifest: str = MANIFEST,
    manifest_sha256: str = MANIFEST_SHA256,
    allowed_topologies: set[tuple[str, int]] | None = None,
) -> None:
    jobs_by_lr = JOBS if jobs_by_lr is None else jobs_by_lr
    allowed_topologies = allowed_topologies or {("80G4-A100", -1)}
    if loss_profile not in {"state_path", "state_kd"}:
        raise ValueError("Unsupported full-Eedi loss profile")
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

    expected = set(jobs_by_lr) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(jobs_by_lr):
        raise ValueError("Expected job-set drift")
    jobs = config.get("jobs") or []
    if len(jobs) != len(expected) or {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("Job-set drift")
    commands = []
    for job in jobs:
        name = str(job.get("name"))
        topology = (str(job.get("sku")), int(job.get("process_count_per_node", 0)))
        if (
            topology not in allowed_topologies
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        observed = {
            "_AZUREML_SINGULARITY_JOB_UAI": str(env.get("_AZUREML_SINGULARITY_JOB_UAI")),
            "VENV_ROOT": str(env.get("VENV_ROOT")),
            "BRANCH": str(env.get("BRANCH")),
            "LOSS_PROFILE": str(env.get("LOSS_PROFILE")),
            "SEGMENT_ROWS": str(env.get("SEGMENT_ROWS")),
            "LEARNING_RATE": str(env.get("LEARNING_RATE")),
            "EXPECTED_CAUSAL_PAIRS": str(env.get("EXPECTED_CAUSAL_PAIRS")),
            "DIAGNOSTIC_STOP_AFTER_ROUND": str(env.get("DIAGNOSTIC_STOP_AFTER_ROUND")),
            "MANIFEST": str(env.get("MANIFEST")),
            "MANIFEST_SHA256": str(env.get("MANIFEST_SHA256")),
        }
        required = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": venv_root,
            "BRANCH": branch,
            "LOSS_PROFILE": loss_profile,
            "SEGMENT_ROWS": segment_rows,
            "LEARNING_RATE": jobs_by_lr[name],
            "EXPECTED_CAUSAL_PAIRS": expected_causal_pairs,
            "DIAGNOSTIC_STOP_AFTER_ROUND": diagnostic_stop_after_round,
            "MANIFEST": manifest,
            "MANIFEST_SHA256": manifest_sha256,
        }
        if observed != required:
            raise ValueError(f"Controlled arm drift: {name}: {observed}")
        observed_entrypoint = env.get("TRAIN_ENTRYPOINT")
        if train_entrypoint is None:
            if observed_entrypoint is not None:
                raise ValueError(f"Unexpected training entrypoint: {name}")
        elif str(observed_entrypoint) != train_entrypoint:
            raise ValueError(f"Training entrypoint drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Matched jobs do not share one command")
    if commands[0].strip() != "set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh":
        raise ValueError("Launcher drift")

    root = Path(__file__).resolve().parent
    launcher = (root / "run_layerwise_task_retention_rehearsal.sh").read_text(encoding="utf-8")
    required_launcher = (
        'case "$BRANCH" in eedi_dense|eedi_full|eedi_full_open_t3|equal_t3_t6|equal_t3_t6_open_t3|t3_heavy|staged_t3_t6)',
        'SEGMENT_ROWS="${SEGMENT_ROWS:-1000}"',
        'LEARNING_RATE="${LEARNING_RATE:-5e-7}"',
        '--learning-rate "$LEARNING_RATE" --lr-scheduler constant',
        '--diagnostic-stop-after-round "$DIAGNOSTIC_STOP_AFTER_ROUND"',
        'if [[ "$BRANCH" == eedi_full || "$BRANCH" == eedi_full_open_t3 ]]',
        "build_layerwise_task_retention_manifests.py",
        "--causal-objective-mode specificity_triplet",
        "--save-steps 50 --checkpoint-retention 2",
    )
    if any(fragment not in launcher for fragment in required_launcher):
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
