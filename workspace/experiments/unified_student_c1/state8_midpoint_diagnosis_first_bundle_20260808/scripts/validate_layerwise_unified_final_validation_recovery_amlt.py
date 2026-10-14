#!/usr/bin/env python3
"""Fail-closed AMLT validator for exact-final-checkpoint single-A100 recovery."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-unified-final-validation-recovery-seed42"


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
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(
        r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b",
        rendered,
    ):
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
        or job.get("mpi") is not False
        or int(job.get("process_count_per_node", 0)) != 1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if (
        submit.get("max_attempts") != 2
        or submit.get("max_run_duration_seconds") != 86400
        or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI")
        != "rankfun_aml"
    ):
        raise ValueError("Retry/duration/UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
        "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
        "tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py",
        "cp scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py scripts/layerwise_recurrent_student_model.py",
        "cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py",
        "checkpoint-round-16241",
        "receipt.get('round') == 16241",
        "receipt.get('world_size') == 16",
        "payload.get('round') == 16241",
        "payload.get('world_size') == 16",
        "len(payload.get('history') or []) == 259845",
        "positions == list(range(259845))",
        "'state_token_alias': '<|vision_pad|>'",
        "'layerwise_transition_rank': 256",
        "'causal_objective_mode': 'specificity_triplet'",
        "'causal_repeat_fraction': 0.05",
        "'normalize_causal_exposure': True",
        "'distributed_world_size': 16",
        "'planned_optimizer_steps': 16241",
        "9961f746cb17e6c314e182d3453b6157022fb48b8de44d2e68bc8e87a1131ccd",
        "30f68f54120e5a9ada0a37b1dba2b6480de0f8eaeac98d0b7ed7c627429f9766",
        "8e038c3a60c2e1b420779cd54eb203b12ae5d51544c87a805fe19d47fdbe294e",
        "--state-architecture layerwise",
        "--layerwise-transition-rank",
        "--resume-from-checkpoint",
        "--validation-only-from-distributed-checkpoint",
        "--finalize-complete-distributed-epoch",
        "completion['status']=='complete'",
        "completion['diagnostic'] is False",
        "completion['micro_steps']==259845",
        "completion['optimizer_steps']==16241",
        "single_A100_validation_from_exact_final_distributed_checkpoint",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("Final validation recovery command drift")
    forbidden = (
        "--distributed-mpi",
        "optimizer.step()",
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "checkpoint-round-16240",
    )
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Distributed/secret/local/checkpoint drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
