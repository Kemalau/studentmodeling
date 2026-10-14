#!/usr/bin/env python3
"""Fail-closed AMLT validator for read-only State8 final-fit milestones."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ROUNDS = {3249, 6497, 9745, 12993}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
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
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if str(code.get("local_dir")) != "$CONFIG_DIR/..":
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    full = {f"state8-finalfit-milestone-r{round_index}" for round_index in ROUNDS}
    if names != (expected_jobs or full):
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"]); round_index = int(name.rsplit("r", 1)[1])
        if round_index not in ROUNDS:
            raise ValueError("Milestone round drift")
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        if env != {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-finalfit-milestone-venv",
            "MILESTONE_VARIANT": "t3_heavy",
            "MILESTONE_ROUND": str(round_index),
        }:
            raise ValueError("Milestone environment drift")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError("Retry/duration drift")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if "bash scripts/run_state8_finalfit_milestone_eval.sh" not in command:
            raise ValueError("Launcher drift")
    launcher = (Path(__file__).resolve().parent / "run_state8_finalfit_milestone_eval.sh").read_text()
    required = (
        "checkpoint-round-$MILESTONE_ROUND",
        "receipt=={'status':'complete','round':expected_round,'world_size':16,'sha256':observed}",
        "list(range(expected_rows))",
        "--validation-only-from-distributed-checkpoint",
        "--resume-from-checkpoint \"$CHECKPOINT/recurrent_state.pt\"",
        "--max-samples-per-group 25",
        "--t3-natural-schema",
        "--num-pairs 150",
        "persistent_history_kv_cache",
        "source_checkpoint_sha256",
        "evaluated_artifact_sha256",
    )
    if any(fragment not in launcher for fragment in required):
        raise ValueError("Milestone launcher invariant drift")
    forbidden = ("--distributed-mpi", "optimizer.step()", "password", "account_key", "access_token")
    if any(fragment.lower() in launcher.lower() for fragment in forbidden):
        raise ValueError("Milestone launcher performs a forbidden operation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
