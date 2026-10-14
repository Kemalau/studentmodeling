#!/usr/bin/env python3
"""Fail-closed AMLT validator for promoted exact-final single-A100 finalization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-task-retention-promoted-final-validation-seed42"
LAUNCHER = "run_layerwise_task_retention_promoted_final_validation.sh"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
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
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if (
        submit.get("max_attempts") != 2
        or submit.get("max_run_duration_seconds") != 86400
        or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI")
        != "rankfun_aml"
        or (submit.get("env") or {}).get("VENV_ROOT")
        != "/tmp/studentmodeling-promoted-finalize-venv"
    ):
        raise ValueError("Retry/duration/UAI/venv drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if command.strip() != f"set -euo pipefail\nbash scripts/{LAUNCHER}":
        raise ValueError("Finalizer launcher drift")

    script = (Path(__file__).resolve().parent / LAUNCHER).read_text(encoding="utf-8")
    required = (
        'VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-promoted-finalize-venv}"',
        "checkpoint-round-16241",
        "checkpoint-round-750",
        "eedi_full_open_t3__state_path__lr2e-06",
        "selected_learning_rate'])==2e-6",
        "selected_causal_weight'])==2.0",
        "receipt=={'status':'complete','round':16241,'world_size':16,'sha256':actual}",
        "payload.get('round')==16241",
        "len(history)==259845",
        "list(range(259845))",
        "len(payload.get('causal_schedule') or {})==12992",
        "promoted_real_order_manifest_sha256",
        "payload['real_order']==manifest['real_order']",
        "audit_layerwise_distributed_checkpoint.py",
        "--expected-round 750",
        "promoted-final-checkpoint-audit.json",
        "promoted-source-checkpoint-audit.json",
        "train_recurrent_student_m4_promoted_open_t3.py",
        "--state-architecture layerwise --layerwise-operation-checkpointing",
        "--layerwise-transition-rank 256",
        "--learning-rate 2e-6",
        "--behavior-kd-weight 0.0",
        "--direct-replay-weight 0.0",
        "--causal-weight 2.0",
        "--causal-objective-mode specificity_triplet",
        "--resume-from-checkpoint \"$CHECKPOINT/recurrent_state.pt\"",
        "--validation-only-from-distributed-checkpoint --finalize-complete-distributed-epoch",
        "completion['micro_steps']==259845",
        "completion['optimizer_steps']==16241",
        "single_A100_validation_from_exact_promoted_final_checkpoint",
        "promoted_final_fit_receipt.json",
        "promoted_open_t3_training_view.json",
        "source_checkpoint_audit.json",
        "final_checkpoint_audit.json",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
    )
    if any(fragment not in script for fragment in required):
        raise ValueError("Promoted finalizer command drift")
    forbidden = (
        "--distributed-mpi",
        "optimizer.step()",
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "checkpoint-round-16240",
        "--diagnostic-warm-start-checkpoint",
    )
    if any(fragment.lower() in script.lower() for fragment in forbidden):
        raise ValueError("Distributed/secret/local/checkpoint drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
