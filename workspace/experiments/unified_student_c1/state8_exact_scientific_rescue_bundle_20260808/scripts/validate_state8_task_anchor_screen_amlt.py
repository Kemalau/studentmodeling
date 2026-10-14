#!/usr/bin/env python3
"""Fail-closed preflight for matched State8 task-anchor Feeds screens."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "state8-task-anchor-base-seed56": ("task_anchor_base_v1", "56", "0.50", "2.0"),
    "state8-task-anchor-c4strong-seed57": (
        "task_anchor_c4strong_v1",
        "57",
        "1.0",
        "4.0",
    ),
}
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-task-anchor-venv",
    "REPAIR_ARM": "taskanchor",
    "REQUESTED_WORLD_SIZE": "4",
    "GRADIENT_ACCUMULATION_STEPS": "1",
    "STATE8_REPAIR_SAVE_STEPS": "5",
    "STATE8_REPAIR_RECOVERY_SAVE_STEPS": "1",
    "TRAIN_ENTRYPOINT": "scripts/train_state8_task_anchored_causal_repair.py",
    "STUDENT_ROOT": "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1",
    "STATE8_T3_ANCHOR_SCHEDULE": "1.0,0.5,0.25",
    "STATE8_OTHER_ANCHOR_WEIGHT": "0.1",
    "STATE8_CAUSAL_LABEL_WEIGHT": "4.0",
    "STATE8_REPAIR_LABEL_WEIGHT": "4.0",
    "STATE8_RANDOM_NEUTRAL": "1",
    "STATE8_GEOMETRY_WEIGHT": "0.10",
    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
    "STATE8_DIFFERENT_MAX_COSINE": "0.90",
    "STATE8_CAUSAL_DIAGNOSIS_ONLY": "0",
    "STATE8_CAUSAL_MARGIN": "0.5",
    "STATE8_LAYERWISE_TRANSITION_ONLY": "0",
    "STATE8_UPDATER_ONLY": "0",
    "STATE8_FORMAL_SCOPE": "unified",
    "STATE8_REPAIR_MANIFEST": "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json",
    "STATE8_REPAIR_MANIFEST_SHA256": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
}
HASHES = {
    "scripts/train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "scripts/run_state8_stateonly_repair_ab.sh": "3b1ddc924ba094fd412d962df3c510fd26ba0d23425e4b7d92165148a41555b8",
    "scripts/train_state8_task_anchored_causal_repair.py": "b622a62fffef45a19b3b02b771cee4412d5f9d17f1a3fe99c6ba046e6111e66b",
    "scripts/train_state8_taxonomy_causal_repair.py": "26e62969a03aab89047dd42d5bbe6d144eae9bb77d0fa0b962b0da89b9b75347",
    "scripts/state8_taxonomy_contract.py": "1ca32112e76c795502f3cb1edbedcc7220f093c133fae8d20ff5cab789e512ed",
    "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    path: Path,
    *,
    bundle: Path | None = None,
    expected_jobs: set[str] | None = None,
) -> None:
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
    expected_names = expected_jobs or set(JOBS)
    if not expected_names or not expected_names.issubset(JOBS):
        raise ValueError("Unknown expected job")
    if {str(job.get("name")) for job in jobs} != expected_names:
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku"),
            job.get("process_count_per_node"),
            job.get("mpi"),
            job.get("sla_tier"),
        ) != ("80G4-A100", -1, True, "Standard"):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry drift: {name}")
        tag, seed, repeat, causal_weight = JOBS[name]
        expected = {
            **COMMON,
            "RUN_TAG": tag,
            "TRAIN_SEED": seed,
            "STATE8_CAUSAL_REPEAT_FRACTION": repeat,
            "STATE8_CAUSAL_WEIGHT": causal_weight,
        }
        observed = {key: str(value) for key, value in (submit.get("env") or {}).items()}
        if observed != expected:
            raise ValueError(f"Environment drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_stateonly_repair_ab.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    if bundle is not None:
        for relative, expected in HASHES.items():
            observed = digest(bundle / relative)
            if observed != expected:
                raise ValueError(f"Bundle hash drift: {relative}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(
        args.config,
        bundle=args.bundle,
        expected_jobs=set(args.jobs) if args.jobs else None,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
