#!/usr/bin/env python3
"""Fail-closed validator for r64 checkpoint-only trajectory finalization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "trajectory-r64-scale-validation-only-seed42"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
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
        or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
    ):
        raise ValueError("Retry/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        "a39d0d1aabe97e7df3bc0424e18a556af338ba6bf4f328ea21b921915bd4727f",
        "1706a01450660c833534e487a8b7aad56a61541ce41cc17ca2c8ed16e2e23d3c",
        "d2a0cf7702f3aaf56bc3629af7b63837f739799f47a3b51e5f5076917116f0b2",
        "checkpoint-round-50",
        "checkpoint_completion.json",
        "receipt.get('round') == 50",
        "payload.get('world_size') == 4",
        "len(payload.get('history') or []) == 200",
        "--state-architecture layerwise",
        "--layerwise-transition-only --layerwise-transition-rank 64",
        "--trajectory-curriculum-only",
        "--trajectory-relative-mse-weight 1.0",
        "--generation-weight 0 --behavior-kd-weight 0",
        "--direct-replay-weight 0 --causal-weight 0 --causal-trajectory-weight 1.0",
        "--diagnostic-stop-after-round 50",
        "--resume-from-checkpoint",
        "--validation-only-from-distributed-checkpoint",
        "completion['micro_steps'] == 200",
        "completion['optimizer_steps'] == 50",
        "recovery_mode",
        "evaluate_recurrent_causal_gate.py",
        "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "--num-pairs 150",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/checkpoints/layerwise_trajectory_curriculum_r64_scale_seed42_n200_20260731",
    )
    if any(value not in command for value in required):
        raise ValueError("Command drift")
    forbidden = (
        "--distributed-mpi",
        "rm -rf \"$$run_dir\"",
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--trainable-projections",
    )
    if any(value.lower() in command.lower() for value in forbidden):
        raise ValueError("Distributed/destructive/secret/local/scalar drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
