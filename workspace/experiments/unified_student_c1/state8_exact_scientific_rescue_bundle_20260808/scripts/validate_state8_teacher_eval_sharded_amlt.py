#!/usr/bin/env python3
"""Fail-closed validation for an N-way Stage-A teacher generation config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    return value


def validate(
    config: dict,
    expected_jobs: set[str] | None = None,
    *,
    num_shards: int = 4,
) -> None:
    if num_shards < 2:
        raise ValueError("Teacher evaluation requires at least two shards")
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(
        r"<placeholder>|<workspace-name>|<your-command>|<job-name>|"
        r"<describe your experiment>|\bREQUIRED\b",
        rendered,
    ):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    jobs = config.get("jobs") or []
    full_set = {
        f"state8-full-history-teacher-eval-shard{index}"
        for index in range(num_shards)
    }
    expected = expected_jobs or full_set
    if not expected or not expected <= full_set:
        raise ValueError(f"Invalid expected job subset: {sorted(expected)}")
    if {str(job.get("name")) for job in jobs} != expected or len(jobs) != len(expected):
        raise ValueError("Job set drift")
    indices = set()
    for job in jobs:
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError("Topology drift")
        env = (job.get("submit_args") or {}).get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError("UAI drift")
        indices.add(int(env["SHARD_INDEX"]))
        command = command_text(job.get("command"))
        validate_shell_syntax(command, str(job["name"]))
        for fragment in (
            "state8_full_history_natural_teacher_v1",
            "ceaf5248707fb793a1090df4a05ce2cbb35cf4e3088db93d6d11bd017bdd842a",
            "state8_teacher_clean_prompt_v2",
            "c35fd46e0cfbccd9141eccb524992bd5452416d32720686f74ad9373f57de314",
            "78850c5434f24c3072b7ab89370400347955686e9aac5b99aadb69ae19c7cfa5",
            "rewrite_state8_teacher_prompts.py",
            "clean_prompt_receipt.json",
            "--max-new-tokens 256",
            "--tasks diagnose_misconception --sources eedi_mining malrule",
            "--save-unredacted-predictions",
            "x['unsharded_filtered_rows']==1494",
        ):
            if fragment not in command:
                raise ValueError(f"Sharded evaluation launcher drift: {fragment}")
        if not re.search(
            rf'--shard-index "\$\$?SHARD_INDEX" --num-shards {num_shards}', command
        ):
            raise ValueError("Sharded evaluation launcher drift: shard arguments")
    expected_indices = {int(name.rsplit("shard", 1)[1]) for name in expected}
    if indices != expected_indices:
        raise ValueError(f"Shard index drift: {indices}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="+")
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    validate(
        load(args.config),
        set(args.jobs) if args.jobs else None,
        num_shards=args.num_shards,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
