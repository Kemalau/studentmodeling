#!/usr/bin/env python3
"""Fail-closed validator for dense bootstrap-to-symmetric rehearsal jobs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_causal_bootstrap_rehearsal import (
    BOOTSTRAP_FRACTION,
    SPECS,
    TRAIN_SHA256,
)


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
    valid_families = (
        {SPECS["bootstrap_sum_w1"]["job"], SPECS["bootstrap_sum_w4"]["job"]},
        {
            SPECS["bootstrap_native25_sum_w1"]["job"],
            SPECS["bootstrap_native25_sum_w4"]["job"],
        },
        {
            SPECS["bootstrap_balanced_sum_w1"]["job"],
            SPECS["bootstrap_balanced_sum_w4"]["job"],
        },
        {
            SPECS["bootstrap_native25_balanced_sum_w1"]["job"],
            SPECS["bootstrap_native25_balanced_sum_w4"]["job"],
        },
        {
            SPECS["donor_directional_sum_w1"]["job"],
            SPECS["donor_directional_sum_w4"]["job"],
        },
    )
    family_match = (
        names in valid_families
        if selected is None
        else bool(names) and any(names.issubset(family) for family in valid_families)
    )
    if names != expected or not family_match or not names.issubset(allowed):
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"])
        branch, spec = allowed[name]
        if (
            job.get("sku") != "80G4-A100"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("BRANCH") != branch
            or str(env.get("CAUSAL_WEIGHT")) != spec["weight"]
        ):
            raise ValueError(f"Retry/UAI/branch drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            TRAIN_SHA256,
            "--causal-preference-reduction sum",
            f"--causal-objective-mode {spec['objective_mode']}",
            f"--causal-bootstrap-fraction {BOOTSTRAP_FRACTION}",
            f"--causal-bootstrap-native-weight {spec['native_weight']}",
            "--causal-bootstrap-donor-weight 1.0",
            "--causal-repeat-fraction 1.0",
            "--diagnostic-stop-after-round 500",
            "--distributed-world-size 4",
            "latest_checkpoint",
            "checkpoint-round-*",
            "sort -V",
            "int(receipt.get('round', -1)) <= 500",
            "int(receipt['round']) % 250 == 0",
            "resume_args=(--resume-from-checkpoint",
            "causal_pairs_seen') == 2000",
            "unique_causal_pairs_seen') == 1726",
            "planned_causal_exposures') == 12000",
            "objective_phase_counts",
            "state_relative_delta_first_500",
            "state_relative_delta_last_500",
            "evaluate_recurrent_causal_gate.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "--num-pairs 150",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "world_model_v1_m3_generation_kd_trainable_seed42_fixedu4_v2_20260730",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
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
