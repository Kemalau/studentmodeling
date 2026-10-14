#!/usr/bin/env python3
"""Fail-closed validator for a frozen M4 checkpoint C4 probe."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
DEFAULT_ROUND = 250


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(
    config: dict,
    checkpoint_round: int = DEFAULT_ROUND,
    *,
    job_name: str | None = None,
    run_dir: str | None = None,
) -> None:
    if checkpoint_round <= 0:
        raise ValueError("Checkpoint round must be positive")
    job_name = job_name or f"m4-checkpoint{checkpoint_round}-c4-seed42"
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|PLACEHOLDER", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {"storage_account_name": "csnewsandfeeds4150361735", "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True}:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != job_name:
        raise ValueError("Job drift")
    job = jobs[0]
    if job.get("sku") != "80G1-A100" or job.get("mpi") is not False or int(job.get("process_count_per_node", 0)) != 1:
        raise ValueError("GPU/MPI drift")
    if job.get("sla_tier") != "Standard":
        raise ValueError("SLA drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    if submit.get("max_attempts") != 2 or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("Retry/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, job_name)
    required = (
        f"checkpoint-round-{checkpoint_round}", "checkpoint_completion.json",
        f"receipt.get('round') == {checkpoint_round}", "world_size') == 4",
        "evaluate_recurrent_causal_gate.py", "c4_stage_a_gate_pairs_seed42_n150_v1.json",
        "--num-pairs 150", "<|vision_pad|>", "diagnostic_only=true",
        f"checkpoint_round={checkpoint_round}",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
    )
    if any(value not in command for value in required):
        raise ValueError("Command drift")
    if run_dir is not None and f"export RUN_DIR={run_dir}" not in command:
        raise ValueError("Run directory drift")
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models", "msrcommonacr.azurecr.io/ml-common-py3.12:latest")
    if any(value in command.lower() for value in forbidden):
        raise ValueError("Secret/local/forbidden path drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round", type=int, default=DEFAULT_ROUND)
    parser.add_argument("--job-name")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    validate(
        load_config(args.config), args.round,
        job_name=args.job_name, run_dir=args.run_dir,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
