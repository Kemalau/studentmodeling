#!/usr/bin/env python3
"""Fail-closed Feeds preflight for matched taxonomy consolidation screens."""

from __future__ import annotations

import argparse
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
    "state8-taxonomy-curriculum-consolidation-t3heavy-seed52": (
        "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1",
        "taxonomy_curriculum_consolidation_t3heavy_v1", "52",
    ),
    "state8-taxonomy-curriculum-consolidation-balanced-seed53": (
        "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable_taxonomy_semantic_v1",
        "taxonomy_curriculum_consolidation_balanced_v1", "53",
    ),
    "state8-taxonomy-curriculum-consolidation-t3heavy-diag-seed54": (
        "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1",
        "taxonomy_curriculum_consolidation_t3heavy_diag_v1", "54",
    ),
    "state8-taxonomy-curriculum-consolidation-balanced-diag-seed55": (
        "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_balanced_curriculum_v1_immutable_taxonomy_semantic_v1",
        "taxonomy_curriculum_consolidation_balanced_diag_v1", "55",
    ),
}
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-taxonomy-consolidation-venv",
    "REPAIR_ARM": "trajectory01",
    "REQUESTED_WORLD_SIZE": "4",
    "GRADIENT_ACCUMULATION_STEPS": "1",
    "TRAIN_ENTRYPOINT": "scripts/train_state8_taxonomy_causal_repair.py",
    "STATE8_CAUSAL_LABEL_WEIGHT": "4.0",
    "STATE8_REPAIR_LABEL_WEIGHT": "4.0",
    "STATE8_RANDOM_NEUTRAL": "1",
    "STATE8_GEOMETRY_WEIGHT": "0.10",
    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
    "STATE8_DIFFERENT_MAX_COSINE": "0.90",
    "STATE8_CAUSAL_DIAGNOSIS_ONLY": "0",
    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
    "STATE8_CAUSAL_WEIGHT": "2.0",
    "STATE8_CAUSAL_MARGIN": "0.5",
    "STATE8_LAYERWISE_TRANSITION_ONLY": "0",
    "STATE8_UPDATER_ONLY": "0",
    "STATE8_FORMAL_SCOPE": "unified",
    "STATE8_REPAIR_MANIFEST": "manifests/layerwise_task_retention_equal_t3_t6_seed42_v1.json",
    "STATE8_REPAIR_MANIFEST_SHA256": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
}


def validate(path: Path, expected_jobs: set[str] | None = None) -> None:
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
        "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    expected_names = expected_jobs or set(JOBS)
    if not expected_names or not expected_names.issubset(JOBS) or {str(job.get("name")) for job in jobs} != expected_names:
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G4-A100", -1, True, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        root, tag, seed = JOBS[name]
        expected = {**COMMON, "STUDENT_ROOT": root, "RUN_TAG": tag, "TRAIN_SEED": seed}
        if "-diag-" in name:
            expected.update({
                "STATE8_GEOMETRY_WEIGHT": "0.05",
                "STATE8_CAUSAL_DIAGNOSIS_ONLY": "1",
                "STATE8_CAUSAL_REPEAT_FRACTION": "1.0",
            })
        if {key: str(value) for key, value in env.items()} != expected:
            raise ValueError(f"Environment drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry drift: {name}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if "run_state8_stateonly_repair_ab.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    launcher = (Path(__file__).resolve().parent / "run_state8_stateonly_repair_ab.sh").read_text()
    for fragment in (
        "STATE8_FORMAL_SCOPE", "state8_qualified_teacher_curriculum_highkd_immutable_v1_taxonomy_semantic_v1",
        "state8_qualified_teacher_balanced_curriculum_immutable_v1_taxonomy_semantic_v1",
        "train_state8_taxonomy_causal_repair.py", "augment_recurrent_causal_diagnosis.py",
        "evaluate_recurrent_diagnosis_c4_taxonomy.py", "strict_state_only",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher invariant drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args(); validate(args.config, set(args.jobs) if args.jobs else None); print(f"validated={args.config}")


if __name__ == "__main__":
    main()
