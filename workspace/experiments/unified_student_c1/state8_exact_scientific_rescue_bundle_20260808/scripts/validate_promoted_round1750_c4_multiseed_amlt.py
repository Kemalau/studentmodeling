#!/usr/bin/env python3
"""Fail-closed validation for the promoted round-1750 multiseed C4 audit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
SEEDS = {
    "promoted-round1750-c4-n150-seed13": (
        "13",
        "9856e643848d11a834761e3d28fb078f6eb70fa6c42fc468c3eb1b3a1aafd693",
    ),
    "promoted-round1750-c4-n150-seed42": (
        "42",
        "edcdb934a65252557f6c569eea9ef7a85e5445549deac9fb4829c3f579e1a9b8",
    ),
    "promoted-round1750-c4-n150-seed97": (
        "97",
        "15ed710992abd71c37bdfa2246a5a38b5eb422e70ecbfb6da933be013d6d9118",
    ),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(
        r"<placeholder>|<workspace-name>|<your-command>|<describe your experiment>",
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
    observed = {str(job.get("name")) for job in jobs}
    expected = set(SEEDS) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(SEEDS) or observed != expected:
        raise ValueError("Job-set drift")
    commands: list[str] = []
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G1-A100"
            or job.get("mpi") is not False
            or int(job.get("process_count_per_node", 0)) != 1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            submit.get("max_attempts") != 2
            or submit.get("max_run_duration_seconds") != 21600
        ):
            raise ValueError(f"Retry/duration drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if (
            str(env.get("PAIR_SEED")),
            str(env.get("PAIR_MANIFEST_SHA256")),
        ) != SEEDS[name]:
            raise ValueError(f"Pair-seed manifest drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("C4 seed jobs do not share one controlled command")
    command = commands[0].replace("$$", "$")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--state-architecture scalar",
        "--num-pairs 50",
        "checkpoint-round-1000",
    )
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar/n50/wrong-checkpoint path leaked")
    required = (
        "checkpoint-round-1750",
        "/mnt/amulet/studentmodeling/eval_artifacts/promoted_round1750_frozen_v1/checkpoint-round-1750",
        "/mnt/amulet/studentmodeling/eval_progress/promoted_round1750_frozen_v1/",
        "FAILED line=%s rc=%s command=%s",
        "audit_layerwise_distributed_checkpoint.py",
        "--expected-round 1750",
        "audit_layerwise_promoted_checkpoint.py",
        "--state-architecture layerwise",
        "--num-pairs 150",
        "c4_stage_a_gate_pairs_seed${PAIR_SEED}_n150_v1.json",
        '--progress-dir "$C4_PROGRESS_DIR"',
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
        "tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py",
        "all_pairs_n']==150",
        "'checkpoint_round':1750",
        "'pair_seed':seed",
    )
    if any(fragment not in command for fragment in required):
        raise ValueError("C4 multiseed command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
