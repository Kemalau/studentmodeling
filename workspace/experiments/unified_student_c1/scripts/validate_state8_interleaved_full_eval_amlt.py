#!/usr/bin/env python3
"""Fail-closed validation for the frozen 20-job interleaved full evaluation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
from materialize_state8_interleaved_full_eval_amlt import (
    C4_SHA,
    FORMAL_SHA,
    IMAGE,
    PAIR_DIGESTS,
    PROGRESS,
    RUN_DIR,
    WORKSPACE,
)


FORMAL = {f"state8-interleaved-full-formal-shard{i:02d}": i for i in range(16)}
MALRULE = {f"state8-interleaved-full-malrule-c4-seed{seed}": seed for seed in PAIR_DIGESTS}
EEDI = "state8-interleaved-full-eedi-c4-final150"


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b|ml-common-py3\.12:latest", rendered):
        raise ValueError("unresolved or forbidden config content")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("storage drift")
    full = set(FORMAL) | set(MALRULE) | {EEDI}
    expected = full if expected_jobs is None else expected_jobs
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != expected or not expected.issubset(full):
        raise ValueError("job-set drift")
    for job in jobs:
        name = str(job["name"])
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if (
            job.get("sku") != "80G1-A100"
            or job.get("process_count_per_node") != 1
            or job.get("mpi") is not False
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"topology drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 259200:
            raise ValueError(f"attempt drift: {name}")
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        if env.get("STATE8_EVAL_RUN_DIR") != RUN_DIR:
            raise ValueError(f"run-dir drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if name in FORMAL:
            if set(env) != {
                "_AZUREML_SINGULARITY_JOB_UAI",
                "VENV_ROOT",
                "STATE8_EVAL_RUN_DIR",
                "NUM_SHARDS",
                "SHARD_INDEX",
            }:
                raise ValueError(f"formal env drift: {name}")
            if env.get("NUM_SHARDS") != "16" or env.get("SHARD_INDEX") != str(FORMAL[name]):
                raise ValueError(f"formal coordinate drift: {name}")
            if FORMAL_SHA not in command or "run_state8_interleaved_full_formal_shard.sh" not in command:
                raise ValueError(f"formal command drift: {name}")
        elif name in MALRULE:
            seed = MALRULE[name]
            if env.get("C4_KIND") != "malrule" or env.get("PAIR_SEED") != str(seed):
                raise ValueError(f"C4 coordinate drift: {name}")
            if env.get("PAIR_MANIFEST_SHA256") != PAIR_DIGESTS[seed]:
                raise ValueError(f"C4 pair digest drift: {name}")
            if env.get("STATE8_EVAL_PROGRESS_NAMESPACE") != PROGRESS:
                raise ValueError(f"C4 progress drift: {name}")
            if C4_SHA not in command or "run_state8_interleaved_full_c4.sh" not in command:
                raise ValueError(f"C4 command drift: {name}")
        else:
            if env.get("C4_KIND") != "eedi" or env.get("STATE8_EVAL_PROGRESS_NAMESPACE") != PROGRESS:
                raise ValueError("Eedi C4 drift")
            if "PAIR_SEED" in env or "PAIR_MANIFEST_SHA256" in env:
                raise ValueError("Eedi C4 has MalRule coordinates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(yaml.safe_load(args.config.read_text()), set(args.jobs) if args.jobs else None)
    print("valid=true")


if __name__ == "__main__":
    main()

