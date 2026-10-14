#!/usr/bin/env python3
"""Fail-closed AMLT validation for the M2 -> M3 C4 transition audit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m2_m3_c4_transition import CONDITIONS


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
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
    expected = {spec["job"]: (condition, spec) for condition, spec in CONDITIONS.items()}
    jobs = config.get("jobs") or []
    selected = expected_jobs if expected_jobs is not None else set(expected)
    if not selected or not selected.issubset(expected):
        raise ValueError(f"Unknown or empty expected job set: {sorted(selected)}")
    if {str(job.get("name")) for job in jobs} != selected:
        raise ValueError("Transition job set drift")
    for job in jobs:
        name = str(job["name"])
        condition, spec = expected[name]
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("CONDITION") != condition
            or env.get("ADAPTER_DIR_REMOTE") != spec["adapter"]
            or env.get("STATE_PATH_REMOTE") != spec["state"]
            or env.get("SOURCE_COMPLETION_REMOTE") != spec["completion"]
        ):
            raise ValueError(f"Identity or remote input drift in {name}")
        if not all(str(env[key]).startswith("/mnt/amulet/studentmodeling/") for key in (
            "ADAPTER_DIR_REMOTE", "STATE_PATH_REMOTE", "SOURCE_COMPLETION_REMOTE"
        )):
            raise ValueError(f"Non-remote model input in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            "export BASE_DATA=/mnt/amulet/studentmodeling/c4_fullopen_20260724/malrule_c1_clean",
            'mkdir -p "$HF_HUB_CACHE" "$AMLT_OUTPUT_DIR/eval/c4"',
            "evaluate_recurrent_causal_gate.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
            "<|vision_pad|>",
            'adapter_model.safetensors',
            'transition_provenance.txt',
        )
        # The evaluator itself records the raw-history/KV-cache invariants; the
        # command must invoke it with the immutable frozen manifest.
        normalized_command = command.replace("$$", "$")
        if any(value not in normalized_command for value in required):
            raise ValueError(f"C4 command drift in {name}")
        forbidden = (
            "export RECEIPT=",
            "password",
            "account_key",
            "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Secret/local/dynamic-receipt drift in {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
