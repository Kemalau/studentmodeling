#!/usr/bin/env python3
"""Fail-closed AMLT validator for diagnosis-first Teacher full evaluation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
PREFIX = "state8-diagnosis-first-teacher-full-eval-shard"


def validate(path: Path, expected: set[str] | None = None) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True,
    }:
        raise ValueError("Storage drift")
    full = {f"{PREFIX}{index:02d}" for index in range(16)}
    wanted = expected or full
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != wanted or not wanted <= full:
        raise ValueError("Job set drift")
    indices = set()
    for job in jobs:
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
            raise ValueError("Retry/duration drift")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError("UAI drift")
        indices.add(int(env["SHARD_INDEX"]))
        command = command_text(job.get("command")); validate_shell_syntax(command, str(job["name"]))
        for fragment in (
            "state8_diagnosis_first_full_history_teacher_v1",
            "semantic_teacher_gate_pass", "diagnosis_evidence_label_v1",
            "add_state8_teacher_markers.py", "--num-shards 16",
            "--tasks diagnose_misconception --sources eedi_mining malrule",
            "--max-new-tokens 256", "--save-unredacted-predictions",
            "marker_contains_sample_information", "m['unsharded_filtered_rows']==1494",
        ):
            if fragment not in command:
                raise ValueError(f"Command contract drift: {fragment}")
    expected_indices = {int(name[-2:]) for name in wanted}
    if indices != expected_indices:
        raise ValueError("Shard index drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+")
    args = parser.parse_args()
    validate(args.config, set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
