#!/usr/bin/env python3
"""Fail-closed preflight for state8 diagnostic single-GPU finalization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-diagnostic-finalizer-seed42"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(path)
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
    if (config.get("storage") or {}).get("student_store") != {
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
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("mpi") is not False
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}; env = submit.get("env") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry drift")
    if env != {"_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml", "VENV_ROOT": "/tmp/studentmodeling-state8-finalizer-venv"}:
        raise ValueError("UAI/venv drift")
    command = command_text(job.get("command")); validate_shell_syntax(command, JOB)
    if "bash scripts/run_state8_diagnostic_finalizer.sh" not in command:
        raise ValueError("Launcher drift")
    script = (Path(__file__).resolve().parent / "run_state8_diagnostic_finalizer.sh").read_text()
    required = (
        "checkpoint-round-3250",
        "receipt=={'status':'complete','round':3250,'world_size':4,'sha256':observed}",
        "len(history)==1000",
        "list(range(12000,13000))",
        "--state-token-count 8",
        "--history-dropout-curriculum --dual-path-curriculum",
        "--online-kd-weight 0.5",
        "--causal-objective-mode specificity_triplet",
        "--validation-only-from-distributed-checkpoint",
        "--resume-from-checkpoint \"$CHECKPOINT/recurrent_state.pt\"",
        "completion.json",
        "evaluate_recurrent_causal_gate.py",
        "evaluate_recurrent_formal_protocols.py",
    )
    if any(fragment not in script for fragment in required):
        raise ValueError("Finalizer invariant drift")
    forbidden = ("--distributed-mpi", "optimizer.step()", "password", "account_key", "access_token")
    if any(fragment.lower() in script.lower() for fragment in forbidden):
        raise ValueError("Finalizer forbidden operation")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); args=parser.parse_args()
    validate(load(args.config)); print(f"validated={args.config}")


if __name__ == "__main__":
    main()
