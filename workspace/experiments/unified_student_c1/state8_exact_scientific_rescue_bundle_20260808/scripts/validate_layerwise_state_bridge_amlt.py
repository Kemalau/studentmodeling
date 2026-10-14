#!/usr/bin/env python3
"""Fail-closed AMLT validator for the zero-training layerwise bridge."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_layerwise_state_bridge_c4 import SPECS


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [
        index
        for marker in ("description:", "target:")
        if (index := text.find(marker)) >= 0
    ]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts) :])
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    expected = set(SPECS.values())
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("Job set drift")
    reverse = {name: mode for mode, name in SPECS.items()}
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G1-A100"
            or job.get("sla_tier") != "Standard"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
        ):
            raise ValueError(f"GPU/MPI/SLA drift in {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
            or env.get("POSITION_MODE") != reverse[name]
        ):
            raise ValueError(f"Retry/UAI/mode drift in {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        required = (
            "evaluate_layerwise_state_bridge.py",
            "test_layerwise_state_bridge.py",
            "--num-pairs 150",
            "--position-mode",
            "POSITION_MODE",
            "c4_stage_a_gate_pairs_seed42_n150_v1.json",
            "state_curriculum_stage_a_cf1_alias_visionpad_seed42_20260727",
            "/mnt/amulet/studentmodeling/models/Qwen3-32B",
            "optimizer_steps=0",
            "decoder_view=state_plus_current_query_strict_late_fusion",
        )
        forbidden = (
            "train_c1_lora.py",
            "train_recurrent_student",
            "password",
            "account_key",
            "access_token",
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
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
