#!/usr/bin/env python3
"""Fail-closed AMLT preflight for completed T3-heavy State8 finalization."""

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
JOB = "state8-taxonomy-t3heavy-finalizer-seed42"


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
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    topology = (
        job.get("sku"),
        job.get("process_count_per_node"),
        job.get("mpi"),
        job.get("sla_tier"),
    )
    if topology != ("80G1-A100", 1, False, "Standard"):
        raise ValueError("Topology drift")
    env = ((job.get("submit_args") or {}).get("env") or {})
    expected = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "STATE8_REPLAY_VARIANT": "t3_heavy",
        "STATE8_KD_VARIANT": "highkd",
        "STATE8_CURRICULUM_NAMESPACE": "taxonomy_semantic_v1",
        "STATE8_FINALIZE_ONLY": "1",
        "TRAIN_ENTRYPOINT": "scripts/train_state8_taxonomy_teacher_curriculum.py",
        "FORMAL_EVALUATOR": "scripts/evaluate_recurrent_formal_protocols_taxonomy.py",
        "STATE8_CURRICULUM_LABEL_WEIGHT": "4.0",
        "STATE8_SAVE_STEPS": "50",
        "TRAIN_SEED": "42",
    }
    for key, value in expected.items():
        if str(env.get(key)) != value:
            raise ValueError(f"Environment drift: {key}")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
        raise ValueError("Retry drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if "bash scripts/run_state8_qualified_teacher_curriculum.sh" not in command:
        raise ValueError("Launcher drift")
    launcher = (
        Path(__file__).resolve().parent / "run_state8_qualified_teacher_curriculum.sh"
    ).read_text(encoding="utf-8")
    required = (
        'STATE8_FINALIZE_ONLY="${STATE8_FINALIZE_ONLY:-0}"',
        'test "$STATE8_REPLAY_VARIANT" = t3_heavy',
        'test "$STATE8_KD_VARIANT" = highkd',
        "assert int(payload.get('world_size',-1))==4",
        "assert int(payload.get('round',-1))==int(sys.argv[2])",
        "--validation-only-from-distributed-checkpoint --finalize-complete-distributed-epoch",
        'test -n "$resume_checkpoint"',
    )
    if any(fragment not in launcher for fragment in required):
        raise ValueError("Finalizer launcher invariant drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
