#!/usr/bin/env python3
"""Fail-closed validation for the isolated G1 bootstrap sensitivity."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from render_m4_causal_bootstrap_g1_sensitivity import TARGET_JOB
from render_m4_causal_bootstrap_rehearsal import TRAIN_SHA256


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    starts = [i for marker in ("description:", "target:") if (i := text.find(marker)) >= 0]
    if not starts:
        raise ValueError("No AMLT YAML document")
    value = yaml.safe_load(text[min(starts):])
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != TARGET_JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    env = (job.get("submit_args") or {}).get("env") or {}
    if (
        job.get("sku") != "80G1-A100"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
        or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
        or env.get("BRANCH") != "bootstrap_sum_w1_g1_sensitivity"
        or str(env.get("CAUSAL_WEIGHT")) != "1.0"
    ):
        raise ValueError("GPU/UAI/branch drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, TARGET_JOB)
    required = (
        TRAIN_SHA256,
        "world_model_v1_m4_causal_bootstrap_sum_w1_g1_sensitivity_seed42_20260731",
        "--diagnostic-max-micro-steps 2000",
        "--gradient-accumulation-steps 4",
        "--save-steps 500",
        "--causal-objective-mode bootstrap_then_symmetric",
        "--causal-bootstrap-fraction 0.25",
        "--causal-preference-reduction sum",
        "checkpoint-*",
        "latest_checkpoint",
        "optimizer_steps')) == 500",
        "causal_pairs_seen') == 2000",
        "unique_causal_pairs_seen') == 1726",
        "single_a100_accum4_sensitivity",
        "abs(float(config.get('causal_bootstrap_fraction')) - 0.25)",
        "evaluate_recurrent_causal_gate.py",
        "--num-pairs 150",
    )
    forbidden = (
        "--distributed-mpi",
        "--distributed-world-size",
        "--diagnostic-stop-after-round",
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
    )
    if any(value not in command for value in required):
        raise ValueError("Command drift")
    if any(value in command.lower() for value in forbidden):
        raise ValueError("Forbidden command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
