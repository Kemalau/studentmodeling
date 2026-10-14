#!/usr/bin/env python3
"""Fail-closed AMLT validation for the query-aligned State8 repair."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
SPECS = {
    "state8-current-query-updater-mse1-seed3072": {
        "runner": "run_state8_current_query_causal_repair.sh",
        "venv": "/tmp/studentmodeling-state8-current-query-repair-venv",
    },
    "state8-eedi-current-query-updater-mse1-seed3072": {
        "runner": "run_state8_eedi_current_query_causal_repair.sh",
        "venv": "/tmp/studentmodeling-state8-eedi-current-query-repair-venv",
    },
    "state8-eedi-same-question-updater-mse1-seed3072": {
        "runner": "run_state8_eedi_same_question_causal_repair.sh",
        "venv": "/tmp/studentmodeling-state8-eedi-same-question-repair-venv",
        "mode": "updater_mse1",
    },
    "state8-eedi-same-question-full-mse1-seed3071": {
        "runner": "run_state8_eedi_same_question_causal_repair.sh",
        "venv": "/tmp/studentmodeling-state8-eedi-same-question-full-venv",
        "mode": "full_mse1",
    },
    "state8-eedi-same-question-full-mse2-seed3073": {
        "runner": "run_state8_eedi_same_question_causal_repair.sh",
        "venv": "/tmp/studentmodeling-state8-eedi-same-question-full-mse2-venv",
        "mode": "full_mse2",
    },
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|RUNNER_SHA256", rendered):
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
    if len(jobs) != 1 or jobs[0].get("name") not in SPECS:
        raise ValueError("Job matrix drift")
    job = jobs[0]
    job_name = str(job["name"])
    spec = SPECS[job_name]
    if (
        job.get("sku"),
        job.get("process_count_per_node"),
        job.get("mpi"),
        job.get("sla_tier"),
    ) != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry/duration drift")
    expected_env = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": spec["venv"],
        "SOURCE_ANCHOR_MODE": spec.get("mode", "updater_mse1"),
        "EXPECTED_STUDENT_ADAPTER_SHA256": "73b6633759febc6ccddac8e81d8715b512a86c3410cf6d2f160b48610d59c558",
        "EXPECTED_STUDENT_STATE_SHA256": "05dd9377584301293ba319ef7cd3bc93ec64f820c171f51abee593185a495aba",
    }
    if submit.get("env") != expected_env:
        raise ValueError("Environment/artifact drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, job_name)
    runner = str(spec["runner"])
    runner_hash = hashlib.sha256(
        (Path(__file__).resolve().parent / runner).read_bytes()
    ).hexdigest()
    if runner_hash not in command or f"bash scripts/{runner}" not in command:
        raise ValueError("Runner binding drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
