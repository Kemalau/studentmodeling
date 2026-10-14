#!/usr/bin/env python3
"""Fail-closed validation for the two-arm post-training semantic C4 screen."""

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
ROOT = "/mnt/amulet/studentmodeling/diagnostics/state8_qualified_teacher_curriculum_highkd_v1_immutable_taxonomy_semantic_v1_strong1_widen8_"
EXPECTED = {
    "state8-strong1-widen8-eval-lr5e7-seed58": ("lr5e7_seed58", ROOT + "lr5e7_seed58"),
    "state8-strong1-widen8-eval-lr1e6-seed59": ("lr1e6_seed59", ROOT + "lr1e6_seed59"),
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
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != set(EXPECTED):
        raise ValueError("Job drift")
    for job in jobs:
        name = str(job["name"])
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        slug, root = EXPECTED[name]
        expected_env = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-strong1-widen8-eval-venv",
            "CANDIDATE_SLUG": slug,
            "STATE8_EVAL_ROOT": root,
        }
        for key, value in expected_env.items():
            if str(env.get(key)) != value:
                raise ValueError(f"Environment drift: {name}/{key}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_strong1_widen8_eval.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    launcher = Path(__file__).with_name("run_state8_strong1_widen8_eval.sh").read_text(encoding="utf-8")
    for fragment in (
        "--num-pairs 150",
        "--num-pairs 50",
        "evaluate_recurrent_diagnosis_c4_taxonomy.py",
        "raw_history_visible_to_decoder",
        "semantic_judgment_pending",
        "exact_label_metric_authoritative",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()

