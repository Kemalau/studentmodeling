#!/usr/bin/env python3
"""Fail-closed preflight for the eight-pair causal-path overfit diagnostic."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "m4-pair8-overfit-mean-seed42": ("pair8_mean", "mean"),
    "m4-pair8-overfit-sum-seed42": ("pair8_sum", "sum"),
}


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [index for marker in ("description:", "target:") if (index := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
    return value


def validate(config: dict, selected: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
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
    names = {str(job.get("name")) for job in jobs}
    expected = set(JOBS) if selected is None else selected
    if names != expected or not names.issubset(JOBS):
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        branch, reduction = JOBS[name]
        if job.get("sku") != "80G1-A100" or job.get("mpi") is not False:
            raise ValueError(f"GPU/MPI drift in {name}")
        if int(job.get("process_count_per_node", 0)) != 1 or job.get("sla_tier") != "Standard":
            raise ValueError(f"Process/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"Retry/UAI drift in {name}")
        if env.get("BRANCH") != branch or env.get("PREFERENCE_REDUCTION") != reduction:
            raise ValueError(f"Branch/reduction drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            "f03c363cd3614965b495cab73629c8ab79eebce4d8dc72a873e205cc1212f4bc",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
            "splitlines()[:8]",
            "--causal-preference-reduction",
            "--causal-repeat-fraction 1.0",
            "--diagnostic-max-micro-steps 200",
            "completion.get('causal_pairs_seen') == 200",
            "completion.get('unique_causal_pairs_seen') == 8",
            "margin_loss_first_50",
            "margin_loss_last_50",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        reduction_forms = (
            '"$$PREFERENCE_REDUCTION"',
            '"$PREFERENCE_REDUCTION"',
        )
        if not any(value in command for value in reduction_forms):
            raise ValueError(f"Preference-reduction command drift in {name}")
        forbidden = (
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Forbidden path/secret in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
