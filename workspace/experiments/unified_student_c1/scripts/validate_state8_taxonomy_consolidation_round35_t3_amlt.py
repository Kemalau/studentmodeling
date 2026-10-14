#!/usr/bin/env python3
"""Fail-closed AMLT preflight for the immutable round-35 T3 probe."""

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
JOB = "state8-taxonomy-consolidation-round35-t3-probe"


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False).replace("<|vision_pad|>", "")
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
    for fragment in (
        "state8_taxonomy_consolidation_t3heavy_round35_immutable_20260808",
        "receipt=={'round':35",
        "e132c15472ab6796484e3ad70a795c704044e7faa2e464df9a8cb06ff2ec36d5",
        "3f1c2bf7e8890bae082d225f7bb61dcc37dd68c9b9b0acff3661553e3d4247b4",
        "--tasks diagnose_misconception",
        "--sources eedi_mining malrule",
        "--t3-natural-schema",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "sampled_rows']==75",
    ):
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
