#!/usr/bin/env python3
"""Fail-closed validation for the read-only State8 step-2000 evaluation."""

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
ROOT = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_"
    "taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4/checkpoint-2000"
)
JOBS = {
    "state8-strong1-midpoint-step2000-t3": "run_state8_strong1_midpoint_t3.sh",
    "state8-strong1-midpoint-step2000-c4": "run_state8_strong1_midpoint_c4.sh",
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
    storage = (config.get("storage") or {}).get("student_store")
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if {row.get("name") for row in jobs} != set(JOBS) or len(jobs) != 2:
        raise ValueError("Job drift")
    for job in jobs:
        name = job["name"]
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        env = submit.get("env") or {}
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-midpoint-eval-venv",
            "STATE8_CHECKPOINT_ROOT": ROOT,
        }
        if any(str(env.get(key)) != value for key, value in expected.items()):
            raise ValueError(f"Environment drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if JOBS[name] not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    root = Path(__file__).resolve().parent
    t3 = (root / "run_state8_strong1_midpoint_t3.sh").read_text(encoding="utf-8")
    c4 = (root / "run_state8_strong1_midpoint_c4.sh").read_text(encoding="utf-8")
    for fragment in ("checkpoint-2000", "--t3-natural-schema", "sampled_rows']==75", "raw_history_visible_to_decoder"):
        if fragment not in t3:
            raise ValueError(f"T3 launcher contract drift: {fragment}")
    for fragment in ("checkpoint-2000", "--num-pairs 50", "evaluate_recurrent_diagnosis_c4_taxonomy.py", "semantic_judgment_pending"):
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
