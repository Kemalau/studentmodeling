#!/usr/bin/env python3
"""Fail-closed preflight for the taxonomy-aware State8 curriculum."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "state8-taxonomy-teacher-curriculum-seed42": ("t3_heavy", "highkd", "42"),
    "state8-taxonomy-balanced-curriculum-seed43": ("balanced", "base", "43"),
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
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
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") not in JOBS:
        raise ValueError("Job drift")
    job = jobs[0]
    job_name = str(job["name"])
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G4-A100", -1, True, "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    env = submit.get("env") or {}
    expected = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "STATE8_REPLAY_VARIANT": JOBS[job_name][0],
        "STATE8_KD_VARIANT": JOBS[job_name][1],
        "TRAIN_SEED": JOBS[job_name][2],
        "STATE8_CURRICULUM_NAMESPACE": "taxonomy_semantic_v1",
        "TRAIN_ENTRYPOINT": "scripts/train_state8_taxonomy_teacher_curriculum.py",
        "FORMAL_EVALUATOR": "scripts/evaluate_recurrent_formal_protocols_taxonomy.py",
        "STATE8_CURRICULUM_LABEL_WEIGHT": "4.0",
        "STATE8_SAVE_STEPS": "50",
    }
    for key, value in expected.items():
        if str(env.get(key)) != value:
            raise ValueError(f"Environment drift: {key}")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, job_name)
    if "run_state8_qualified_teacher_curriculum.sh" not in command:
        raise ValueError("Entrypoint drift")
    launcher = (Path(__file__).resolve().parent / "run_state8_qualified_teacher_curriculum.sh").read_text()
    for fragment in (
        "taxonomy_semantic_v1",
        "STATE8_CURRICULUM_LABEL_WEIGHT",
        "train_state8_taxonomy_teacher_curriculum.py",
        "evaluate_recurrent_formal_protocols_taxonomy.py",
        "semantic_judge_authoritative",
        "exact_label_metric_authoritative",
        "counts=={0.0:1000,0.25:1000,0.5:1000,0.75:1000,1.0:1000}",
        'STATE8_SAVE_STEPS="${STATE8_SAVE_STEPS:-125}"',
        'RESUME_CADENCE_ARGS=(--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 5)',
        "state8_curriculum_data_cache.py verify",
        "state8_taxonomy_curriculum_data_v1",
        "stage data_cache_verified",
        "STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_v3",
        'ln -s "$DATA_CACHE_ROOT/common/valid.jsonl" "$VALID_JSONL"',
        'ln -s "$DATA_CACHE_ROOT/$STATE8_REPLAY_VARIANT/train5000.jsonl" "$SUBSET"',
        '--train-jsonl "$SUBSET" --valid-jsonl "$VALID_JSONL"',
        '--save-steps "$STATE8_SAVE_STEPS"',
        '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"',
        "--probe-expected-cadence",
        "quarantine_incompatible_checkpoint_receipts.py",
        "incompatible_checkpoint_quarantine.json",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher invariant drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
