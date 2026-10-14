#!/usr/bin/env python3
"""Fail-closed AMLT validator for layerwise full-corpus final fit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-unified-finalfit-seed42"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
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
    if storage != {"storage_account_name": "csnewsandfeeds4150361735", "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True}:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if job.get("sku") != "2x80G8-A100-IB-NvLink" or job.get("mpi") is not True or int(job.get("process_count_per_node", 0)) != -1 or job.get("sla_tier") != "Standard":
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 345600 or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
        raise ValueError("Retry/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440",
        "20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a",
        "7456f167e16c79025ca029d6dccd18612ae3a92c33caede06da995fe6e8be2b1",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
        "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
        "tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py",
        "cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py",
        "cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py",
        "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd",
        "cache_full259845_top256_seed42_v2_16way",
        "layerwise_world_model_v1/selection/seed42.json",
        "layerwise_trajectory_curriculum_",
        "--init-run-dir",
        "SELECTED_RUN_DIR",
        "completion['micro_steps'] == 200",
        "completion['causal_pairs_seen'] == 200",
        "completion['optimizer_steps'] == 50",
        "completion['sha256']['adapter_model']",
        "--state-architecture layerwise",
        "--layerwise-operation-checkpointing",
        "--layerwise-transition-rank",
        "SELECTED_TRANSITION_RANK",
        "--distributed-mpi --distributed-world-size 16",
        "--save-steps 250 --checkpoint-retention 2",
        "select_latest_valid_distributed_checkpoint.py",
        "--expected-world-size 16",
        "--max-round 16241 --save-steps 250",
        "OMPI_COMM_WORLD_LOCAL_RANK:?MPI local rank is required",
        "LOCAL_RANK=0",
        "OMPI_COMM_WORLD_SIZE:?MPI world size is required",
        "--causal-repeat-fraction 0.05 --normalize-causal-exposure",
        "--behavior-kd-weight 0.5",
        "--lr-scheduler cosine --warmup-ratio 0.03",
        "flock -x 7",
        "flock -x 6",
        "/tmp/layerwise-final-data.complete",
        "/tmp/layerwise-final-model.complete",
        "completion['real_examples_seen_once'] is True",
        "completion['all_causal_pairs_covered'] is True",
        "causal_branch_dropout_mode'] == 'eval_deterministic",
        "layerwise_operation_checkpointing'] is True",
        "config['planned_optimizer_steps'] == 16241",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/checkpoints/layerwise_unified_world_model_seed42_full259845_v1_20260731",
    )
    if any(value not in command for value in required):
        raise ValueError("Final-fit command drift")
    if not re.search(
        r'if \[\[ -n "(?:\$\$|\$)selected_checkpoint" \]\]; then', command
    ):
        raise ValueError("Final-fit hash-valid resume guard drift")
    forbidden = ("password", "account_key", "access_token", "/home/v-yuejiali/studentmodeling/models", "--init-recurrent-state", "--trainable-projections", "diagnostic-max")
    if any(value.lower() in command.lower() for value in forbidden):
        raise ValueError("Secret/local/scalar/diagnostic path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
