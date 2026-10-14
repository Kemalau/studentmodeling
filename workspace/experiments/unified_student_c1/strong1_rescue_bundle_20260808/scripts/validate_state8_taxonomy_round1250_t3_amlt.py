#!/usr/bin/env python3
"""Fail-closed preflight for the immutable round-1250 State8 T3 screen."""

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
JOB = "state8-taxonomy-t3heavy-round1250-t3-screen"


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    placeholder_view = rendered.replace("<|vision_pad|>", "")
    if re.search(r"<[^>]+>|\bREQUIRED\b", placeholder_view):
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
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
        raise ValueError("Retry drift")
    if (submit.get("env") or {}) != {"_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml"}:
        raise ValueError("UAI drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    required = (
        "checkpoint-round-1250",
        "receipt['round']==1250",
        "bf75b17f641731c84ac05718d67563c4827cd80eda78d83a2c2b1c94cdde4260",
        "202216966726650d70707678c67d0573f9d0fa3486583fde2c3e0b08c62f8613",
        "--tasks diagnose_misconception",
        "--sources eedi_mining malrule",
        "--t3-natural-schema",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "sampled_rows']==75",
    )
    for fragment in required:
        if fragment not in command:
            raise ValueError(f"Missing invariant: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
