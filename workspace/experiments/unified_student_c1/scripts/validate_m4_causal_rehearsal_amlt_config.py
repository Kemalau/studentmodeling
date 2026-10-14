#!/usr/bin/env python3
"""Fail-closed validator for dense recurrent causal-rehearsal jobs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_causal_rehearsal import SPECS, TRAIN_SHA256


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    allowed = {spec["job"]: (branch, spec) for branch, spec in SPECS.items()}
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    expected = names if selected is None else selected
    if not expected or names != expected or not names.issubset(allowed):
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        branch, spec = allowed[name]
        if job.get("sku") != "80G4-A100" or job.get("mpi") is not True:
            raise ValueError(f"GPU/MPI drift in {name}")
        if int(job.get("process_count_per_node", 0)) != -1:
            raise ValueError(f"Process count drift in {name}")
        if job.get("sla_tier") != "Standard":
            raise ValueError(f"SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2:
            raise ValueError(f"Retry drift in {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift in {name}")
        if env.get("BRANCH") != branch or str(env.get("CAUSAL_WEIGHT")) != spec["weight"]:
            raise ValueError(f"Branch environment drift in {name}")
        if "NORMALIZE_CAUSAL" in env:
            raise ValueError(f"Unexpected normalization environment in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            TRAIN_SHA256,
            "--causal-repeat-fraction 1.0",
            "--diagnostic-stop-after-round 500",
            "--distributed-world-size 4",
            "causal_pairs_seen') == 2000",
            "unique_causal_pairs_seen') == 1726",
            "planned_causal_exposures') == 12000",
            "evaluate_recurrent_causal_gate.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
            f"--causal-preference-reduction {spec['preference_reduction']}",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        forbidden = (
            "$$checkpoint",
            "$checkpoint",
            "checkpoint-round-250",
            "receipt.get('round') == 250",
            "--normalize-causal-exposure",
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Forbidden path/protocol in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
