#!/usr/bin/env python3
"""Fail-closed validator for the two-arm task-retention rehearsal."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "layerwise-task-retention-equal-t3-t6-seed42": (
        "equal_t3_t6",
        "manifests/layerwise_task_retention_equal_t3_t6_seed42_v1.json",
        "32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f",
    ),
    "layerwise-task-retention-t3-heavy-seed42": (
        "t3_heavy",
        "manifests/layerwise_task_retention_t3_heavy_seed42_v1.json",
        "ba8ad97ea29111ce6a5d15f17f3e2bf88fe44a47ef5f10a889815d782f12438e",
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
    code = config.get("code") or {}
    # Source configs retain the symbolic upload root; AMLT dumps replace it
    # with the already-uploaded project storage id and omit local_dir.
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code-upload root drift")
    ignores = set(code.get("ignore") or [])
    if not {"data/", "results/", ".git/"}.issubset(ignores):
        raise ValueError("Code upload can leak large/local artifacts")

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
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("VENV_ROOT") != "/tmp/studentmodeling-layerwise-task-retention-venv":
            raise ValueError(f"Venv drift: {name}")
        observed = (
            str(env.get("BRANCH")),
            str(env.get("MANIFEST")),
            str(env.get("MANIFEST_SHA256")),
        )
        if observed != JOBS[name]:
            raise ValueError(f"Controlled arm drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Task-retention arms do not share one command")
    command = commands[0]
    if command.strip() != "set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh":
        raise ValueError("Task-retention launcher drift")

    script = (Path(__file__).resolve().parent / "run_layerwise_task_retention_rehearsal.sh").read_text(encoding="utf-8")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--state-architecture scalar",
        "formal_evaluation_fixedu4_v2/test.jsonl --train",
    )
    if any(fragment.lower() in script.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar/exact-resume path leaked")
    required = (
        'VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-layerwise-task-retention-venv}"',
        'python() { "$VENV_ROOT/bin/python" "$@"; }',
        "import accelerate,bitsandbytes,peft,safetensors,transformers",
        "checkpoint-round-750",
        "--expected-round 750",
        "x['history_rows']==12000 and x['observed_causal_history_rows']==600",
        "--diagnostic-warm-start-checkpoint \"$CHECKPOINT/recurrent_state.pt\"",
        "select_latest_distributed_checkpoint.py",
        'TRAINER_SHA256="$(sha256sum scripts/train_recurrent_student_m4.py',
        'ENTRYPOINT_SHA256="$(sha256sum "$TRAIN_ENTRYPOINT"',
        'CHECKPOINT_MIRROR_DIR="/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/',
        '--checkpoint-mirror-dir "$CHECKPOINT_MIRROR_DIR"',
        '--resume-from-checkpoint "$RESUME_CHECKPOINT"',
        'DIAGNOSTIC_START_GLOBAL_POSITION="${DIAGNOSTIC_START_GLOBAL_POSITION:-12000}"',
        "--diagnostic-index-manifest \"$MANIFEST\"",
        'DIAGNOSTIC_STOP_AFTER_ROUND="${DIAGNOSTIC_STOP_AFTER_ROUND:-$(((DIAGNOSTIC_START_GLOBAL_POSITION + SEGMENT_ROWS) / 4))}"',
        "--distributed-mpi --distributed-world-size 4",
        "--state-architecture layerwise",
        "--layerwise-operation-checkpointing",
        "--layerwise-transition-rank 256",
        'LEARNING_RATE="${LEARNING_RATE:-5e-7}"',
        '--learning-rate "$LEARNING_RATE" --lr-scheduler constant',
        "--causal-objective-mode specificity_triplet",
        "--causal-repeat-fraction 0.05",
        "--causal-weight 2.0",
        "--normalize-causal-exposure",
        'SEGMENT_ROWS="${SEGMENT_ROWS:-1000}"',
        'EXPECTED_CAUSAL_PAIRS="${EXPECTED_CAUSAL_PAIRS:-$((SEGMENT_ROWS / 20))}"',
        "completion['micro_steps']==segment_rows",
        "completion['optimizer_steps']==segment_rows//4",
        "completion['history_global_position_start']==start",
        "completion['history_global_position_end']==start+segment_rows-1",
        "completion['causal_pairs_seen']==causal_pairs",
        "'exact_resume':False",
        "formal_example_id_overlap",
        "--num-pairs 150",
        "--max-samples-per-group 25",
        "--tasks predict_error_persistence",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/world_model_v1/final_fit/cache_full259845_top256_seed42_v2_16way",
        "WANDB_RUN_ID is required for an isolated diagnostic workspace",
        "${TMPDIR:-/tmp}/layerwise_task_retention_${BRANCH}_${RUN_ID}",
        'test ! -e "$RUN_DIR"',
        "b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440",
        "20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
    )
    if any(fragment not in script for fragment in required):
        raise ValueError("Task-retention command drift")
    if 'rm -rf "$RUN_DIR"' in script:
        raise ValueError("Shared diagnostic workspace cleanup can block on Blob/FUSE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
