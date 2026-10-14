#!/usr/bin/env python3
"""Fail-closed AMLT validator for the grouped layer-transport C4 screen."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_one_shot_layer_transport_c4 import SPECS


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


def validate(config: dict, selected: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    allowed = {spec["job"]: (name, spec) for name, spec in SPECS.items()}
    jobs = config.get("jobs") or []
    names = {str(job.get("name")) for job in jobs}
    expected = selected if selected is not None else set(allowed)
    if not expected or not expected.issubset(allowed) or names != expected:
        raise ValueError(f"Job set drift: {sorted(names)}")
    for job in jobs:
        name = str(job["name"]); group, spec = allowed[name]
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift in {name}")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("TRANSPORT_GROUP") != group
            or env.get("LAYER_GROUPS") != spec["groups"]
        ):
            raise ValueError(f"Retry/UAI/layer-group drift in {name}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        required = (
            "state_swap_misconception_smoke.py", "--allow-manifest-prefix",
            "--num-pairs 50", "--layer-groups", "test_state_swap_layer_groups.py",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B", "diagnostic_only=true",
        )
        forbidden = (
            "train_c1_lora.py", "evaluate_c1.py", "probe_student_state_smoke.py",
            "summarize_stage_a_gate.py", "password", "account_key", "access_token",
            "/home/v-yuejiali/studentmodeling/models",
            "msrcommonacr.azurecr.io/ml-common-py3.12:latest",
        )
        if any(value not in command for value in required):
            raise ValueError(f"Command drift in {name}")
        if any(value in command.lower() for value in forbidden):
            raise ValueError(f"Training/secret/local path leaked into {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
