#!/usr/bin/env python3
"""Fail-closed validator for the layerwise recurrent training gate."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-training-memory-gradient-gate-seed42"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT config: {path}")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
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
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "80G1-A100"
        or job.get("sla_tier") != "Standard"
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    if (
        submit.get("max_attempts") != 2
        or env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
    ):
        raise ValueError("Retry/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        "ab25b40f2b4e91582c9a3bd8bd784161f44aaa6ff5c7aa4c6d19da81693bac32",
        "c3ff9be2eb5dc0216f1be62bd8963901bad36ebe897f83c336fad1274fd64b09",
        "9be3e584cbeb1dc3f1fd7c683106edd547d1b3274934d551d011a4f43e3bcc54",
        "--state-architecture layerwise",
        "--diagnostic-max-micro-steps 4",
        "--gradient-accumulation-steps 1",
        "--causal-objective-mode donor_directional",
        "--causal-preference-reduction sum",
        "--max-updates 4 --train-limit 1200 --valid-limit 300",
        "cf4_balanced_paired_state_swap_train_seed20260731_v1.json",
        "cf4_balanced_paired_state_swap_valid_seed20260801_v1.json",
        "assert completion['contains_causal_pair'] is True",
        "assert gradient['lora']['gradient_abs_sum'] > 0",
        "peak_cuda_reserved_gib",
        "causal_branch_dropout_mode'] == 'eval_deterministic",
        "causal_backward_graph_max_views",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/diagnostics/layerwise_recurrent_training_gate_seed42_20260731",
    )
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--trainable-projections",
        "--init-recurrent-state",
        "--distributed-mpi",
    )
    if any(value not in command for value in required):
        raise ValueError("Layerwise gate command drift")
    if any(value.lower() in command.lower() for value in forbidden):
        raise ValueError("Secret/local/scalar/distributed path leaked into gate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
