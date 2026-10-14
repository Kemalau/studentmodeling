#!/usr/bin/env python3
"""Fail-closed preflight for the isolated pair-8 causal bootstrap job."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_causal_bootstrap_pair8 import SPECS, TRAIN_SHA256


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [index for marker in ("description:", "target:") if (index := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
    return value


def validate(config: dict, variant: str = "donor_only") -> None:
    spec = SPECS[variant]
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
    if len(jobs) != 1 or jobs[0].get("name") != spec["job"]:
        raise ValueError("Job set drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    expected_env = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "BRANCH": spec["branch"],
        "PREFERENCE_REDUCTION": "sum",
        "OBJECTIVE_MODE": spec["objective_mode"],
        "BOOTSTRAP_FRACTION": "0.25",
        "BOOTSTRAP_NATIVE_WEIGHT": spec["native_weight"],
        "BOOTSTRAP_DONOR_WEIGHT": "1.0",
    }
    if submit.get("max_attempts") != 2 or env != expected_env:
        raise ValueError("Retry/UAI/branch environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, spec["job"])
    required = (
        TRAIN_SHA256,
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
        "splitlines()[:8]",
        "--causal-preference-reduction",
        "--causal-objective-mode",
        "--causal-bootstrap-fraction",
        "--causal-bootstrap-native-weight",
        "--causal-bootstrap-donor-weight",
        "--causal-repeat-fraction 1.0",
        "--diagnostic-max-micro-steps 200",
        "completion.get('causal_pairs_seen') == 200",
        "completion.get('unique_causal_pairs_seen') == 8",
        "objective_phase_counts",
        "state_relative_delta_first_50",
        "state_relative_delta_last_50",
    )
    if any(value not in command for value in required):
        raise ValueError("Command drift")
    variable_arguments = {
        "--causal-preference-reduction": "PREFERENCE_REDUCTION",
        "--causal-objective-mode": "OBJECTIVE_MODE",
        "--causal-bootstrap-fraction": "BOOTSTRAP_FRACTION",
        "--causal-bootstrap-native-weight": "BOOTSTRAP_NATIVE_WEIGHT",
        "--causal-bootstrap-donor-weight": "BOOTSTRAP_DONOR_WEIGHT",
    }
    for argument, variable in variable_arguments.items():
        forms = (
            f'{argument} "$${variable}"',
            f'{argument} "${variable}"',
        )
        if not any(form in command for form in forms):
            raise ValueError(f"Command variable drift: {argument}")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
    )
    if any(value in command.lower() for value in forbidden):
        raise ValueError("Forbidden path/secret")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(SPECS), default="donor_only")
    args = parser.parse_args()
    validate(load(args.config), args.variant)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
