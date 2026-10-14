#!/usr/bin/env python3
"""Fail-closed validator for an exact diagnosis-first checkpoint screen."""

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
ROOT_PREFIX = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_"
    "taxonomy_semantic_v1_exact_diagnosis_first_seed64_single_ga4/checkpoint-"
)
FORMAL_SHA = "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9"
C4_SHA = "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b"
ENTRYPOINTS = {
    "t3": "run_state8_exact_diagnosis_midpoint_t3.sh",
    "c4": "run_state8_exact_diagnosis_midpoint_c4.sh",
}
ALLOWED_STEPS = {2000, 3000, 4000, 5000}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store")
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    observed_jobs = {str(job.get("name") or "") for job in jobs}
    if not jobs or len(observed_jobs) != len(jobs):
        raise ValueError("Job set drift")
    observed_steps: set[int] = set()
    for job in jobs:
        name = job["name"]
        match = re.fullmatch(
            r"state8-exact-diagnosis-midpoint-step(2000|3000|4000|5000)-(t3|c4)",
            str(name),
        )
        if match is None:
            raise ValueError("Job set drift")
        step = int(match.group(1)); kind = match.group(2)
        observed_steps.add(step)
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        expected_env = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-exact-midpoint-eval-venv",
            "STATE8_EXACT_CHECKPOINT_ROOT": ROOT_PREFIX + str(step),
            "EXACT_DIAGNOSIS_FORMAL_SHA256": FORMAL_SHA,
            "EXACT_DIAGNOSIS_C4_SHA256": C4_SHA,
        }
        if step != 2000:
            expected_env["STATE8_EXACT_CHECKPOINT_STEP"] = str(step)
        if (submit.get("env") or {}) != expected_env:
            raise ValueError(f"Environment/provenance drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if ENTRYPOINTS[kind] not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    if len(observed_steps) != 1:
        raise ValueError("Mixed checkpoint steps")
    root = Path(__file__).resolve().parent
    t3 = (root / ENTRYPOINTS["t3"]).read_text()
    c4 = (root / ENTRYPOINTS["c4"]).read_text()
    for fragment in ("2000|3000|4000|5000", "evaluate_recurrent_formal_protocols_diagnosis_first.py", "sampled_rows']==75", "strict_state_only"):
        if fragment not in t3:
            raise ValueError(f"T3 launcher contract drift: {fragment}")
    for fragment in (
        "2000|3000|4000|5000",
        "evaluate_recurrent_diagnosis_c4_diagnosis_first.py",
        "parent_manifest_sha256",
        "'selection':'ordered_prefix'",
        "--num-pairs 50",
        "semantic_judgment_pending",
    ):
        if fragment not in c4:
            raise ValueError(f"C4 launcher contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
