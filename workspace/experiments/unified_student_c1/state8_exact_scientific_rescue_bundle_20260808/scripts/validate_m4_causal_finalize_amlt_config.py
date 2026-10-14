#!/usr/bin/env python3
"""Fail-closed validator for single-GPU round-500 causal finalization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
EXPECTED = {
    "m4-causal-finalize-raw4-seed42": ("raw4", "4.0", "0"),
    "m4-causal-finalize-mean-normalized-seed42": ("mean_normalized", "1.0", "1"),
}


def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict, selected_names: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b|PLACEHOLDER", rendered):
        raise ValueError("Unresolved placeholder")
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
    expected = set(EXPECTED) if selected_names is None else selected_names
    if not expected or not expected.issubset(EXPECTED):
        raise ValueError(f"Unknown selected jobs: {sorted(expected)}")
    if {str(job.get("name")) for job in jobs} != expected or len(jobs) != len(expected):
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        if job.get("sku") != "80G1-A100" or job.get("mpi") is not False:
            raise ValueError(f"GPU/MPI drift in {name}")
        if int(job.get("process_count_per_node", 0)) != 1:
            raise ValueError(f"Process-count drift in {name}")
        if job.get("sla_tier") != "Standard":
            raise ValueError(f"SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        branch, weight, normalize = EXPECTED[name]
        if submit.get("max_attempts") != 2 or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"Retry/UAI drift in {name}")
        if env.get("BRANCH") != branch or str(env.get("CAUSAL_WEIGHT")) != weight \
                or str(env.get("NORMALIZE_CAUSAL")) != normalize:
            raise ValueError(f"Branch label drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            "f03c363cd3614965b495cab73629c8ab79eebce4d8dc72a873e205cc1212f4bc",
            "checkpoint-round-500",
            "checkpoint_completion.json",
            "--validation-only-from-distributed-checkpoint",
            "--resume-from-checkpoint",
            "--diagnostic-stop-after-round 500",
            "len(payload['history']) == 2000",
            "causal_pairs_seen') == 288",
            "evaluate_recurrent_causal_gate.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        forbidden = (
            "--distributed-mpi",
            "optimizer.step()",
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Distributed/secret/local/forbidden drift in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load_config(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
