#!/usr/bin/env python3
"""Fail-closed AMLT validator for the exact diagnosis-first State8 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "state8-exact-diagnosis-first-curriculum-c4-seed64"
FILES = {
    "run_state8_exact_diagnosis_pipeline.sh": "24adae367a3161a51d5c274791b4f7c0fbb5b7a323308d9dd572e3a820f021f2",
    "run_state8_exact_diagnosis_curriculum.sh": "27f2f6d473956f1d53acdc6c52bbc2d8264a180f6a0f5ce861cab850a1430aa3",
    "train_state8_diagnosis_first_teacher_curriculum.py": "214e502b44256c309643b377a68af74ae7d4afca3c2e2351cac2b6eb19158a7b",
    "train_state8_diagnosis_first_causal_repair.py": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
    "evaluate_recurrent_formal_protocols_diagnosis_first.py": "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "evaluate_recurrent_diagnosis_c4_diagnosis_first.py": "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b",
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
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
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
        raise ValueError("Retry/duration drift")
    env = submit.get("env") or {}
    expected_env = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-exact-diagnosis-venv",
        "STATE8_DIAGNOSIS_FIRST_CURRICULUM_SHA256": FILES["train_state8_diagnosis_first_teacher_curriculum.py"],
        "STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256": FILES["train_state8_diagnosis_first_causal_repair.py"],
        "STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256": FILES["evaluate_recurrent_formal_protocols_diagnosis_first.py"],
        "STATE8_DIAGNOSIS_FIRST_C4_SHA256": FILES["evaluate_recurrent_diagnosis_c4_diagnosis_first.py"],
    }
    if {key: str(env.get(key)) for key in expected_env} != expected_env:
        raise ValueError("Environment drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if "run_state8_exact_diagnosis_pipeline.sh" not in command:
        raise ValueError("Entrypoint drift")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    pipeline = (root / "run_state8_exact_diagnosis_pipeline.sh").read_text()
    for fragment in (
        "run_state8_exact_diagnosis_curriculum.sh",
        "train_state8_diagnosis_first_causal_repair.py",
        "STATE8_RANDOM_NEUTRAL=1",
        "STATE8_GEOMETRY_WEIGHT=0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT=0.05",
        "STATE8_CAUSAL_REPEAT_FRACTION=0.50",
        "STATE8_FORMAL_SCOPE=unified",
        "REQUESTED_WORLD_SIZE=1",
        "evaluate_recurrent_causal_gate.py",
        "answer_c4_150",
        "--num-pairs 150",
    ):
        if fragment not in pipeline:
            raise ValueError(f"Pipeline contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
