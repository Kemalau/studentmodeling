#!/usr/bin/env python3
"""Fail-closed validator for conditional round-500 causal rescue arms."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "layerwise-c500-rescue-dense-same-total-seed42": (
        "dense_same_total",
        "0.10",
        "1.0",
    ),
    "layerwise-c500-rescue-strong-same-density-seed42": (
        "strong_same_density",
        "0.05",
        "2.0",
    ),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid AMLT YAML: {path}")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
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
    expected = set(JOBS) if expected_jobs is None else expected_jobs
    if not expected or not expected.issubset(JOBS):
        raise ValueError("Expected job-set drift")
    jobs = config.get("jobs") or []
    if len(jobs) != len(expected) or {job.get("name") for job in jobs} != expected:
        raise ValueError("Job-set drift")
    commands = []
    for job in jobs:
        name = str(job.get("name"))
        if (
            job.get("sku") != "80G4-A100"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
            raise ValueError(f"Retry/duration drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        observed = (
            str(env.get("BRANCH")),
            str(env.get("CAUSAL_REPEAT_FRACTION")),
            str(env.get("CAUSAL_WEIGHT")),
        )
        if observed != JOBS[name]:
            raise ValueError(f"Controlled arm drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Rescue arms do not share one controlled command")
    command = commands[0]
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--state-architecture scalar",
        "--resume-from-checkpoint",
        "layerwise_unified_world_model_seed42_full259845_v1_20260731/completion.json",
    )
    if any(fragment.lower() in command.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar/exact-resume path leaked")
    required = (
        "4a9518f50c6840e068bda0848b50ef0332e239d5ed2475b0785f0fc71680561f",
        "fd6510465abafbd897ac4acbf27dd0da14b179637ba9a632f4c5cb539214c5ad",
        "6690ece8935c6343bec7136ad34a9899b72a866c7bde6286ca26b2c82890007f",
        "checkpoint-round-500",
        "audit_layerwise_distributed_checkpoint.py",
        "--expected-round 500",
        "x['checkpoint_round']==500 and x['world_size']==16",
        "x['history_rows']==8000 and x['observed_causal_history_rows']==400",
        "--diagnostic-warm-start-checkpoint \"$CHECKPOINT/recurrent_state.pt\"",
        "--diagnostic-start-global-position 8000",
        "--diagnostic-stop-after-round 2250",
        "--distributed-mpi --distributed-world-size 4",
        "--state-architecture layerwise",
        "--layerwise-operation-checkpointing",
        "--layerwise-transition-rank 256",
        "--learning-rate 5e-7 --lr-scheduler constant",
        "--causal-objective-mode specificity_triplet",
        "--causal-repeat-fraction \"$CAUSAL_REPEAT_FRACTION\"",
        "--causal-weight \"$CAUSAL_WEIGHT\"",
        "--normalize-causal-exposure",
        "diagnostic_weight_only_warm_start",
        "diagnostic_optimizer_reset",
        "completion['micro_steps']==1000",
        "completion['optimizer_steps']==250",
        "completion['history_global_position_start']==8000",
        "completion['history_global_position_end']==8999",
        "config['diagnostic_segment_rounds']==250",
        "'dense_same_total':100,'strong_same_density':50",
        "'exact_resume':False",
        "--state-architecture layerwise",
        "--num-pairs 150",
        "--progress-dir \"$C4_PROGRESS_DIR\"",
        "--progress-key \"$C4_PROGRESS_KEY\"",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way",
        "/mnt/amulet/studentmodeling/diagnostics/layerwise_checkpoint500_causal_rescue_",
        "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
        "tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py",
    )
    normalized_command = command.replace("$$", "$")
    if any(fragment not in normalized_command for fragment in required):
        raise ValueError("Causal-rescue command drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
