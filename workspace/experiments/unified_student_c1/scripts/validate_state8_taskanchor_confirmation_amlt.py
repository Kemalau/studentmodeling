#!/usr/bin/env python3
"""Fail closed on a materialized task-anchor full confirmation config."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
    from materialize_state8_taskanchor_confirmation_amlt import (
        IMAGE,
        PAIR_HASHES,
        WORKSPACE,
    )
except ImportError:  # package-style unit tests
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )
    from scripts.materialize_state8_taskanchor_confirmation_amlt import (
        IMAGE,
        PAIR_HASHES,
        WORKSPACE,
    )


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("image drift")
    storage = (config.get("storage") or {}).get("student_store")
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 19:
        raise ValueError("expected 19 independent jobs")
    formal = [job for job in jobs if "formal-shard" in str(job.get("name"))]
    causal = [job for job in jobs if "c4-seed" in str(job.get("name"))]
    if len(formal) != 16 or len(causal) != 3:
        raise ValueError("job cardinality drift")
    common = None
    for job in jobs:
        if job.get("sku") != "80G1-A100" or job.get("mpi") is not False:
            raise ValueError("GPU topology drift")
        if int(job.get("process_count_per_node", 0)) != 1 or job.get("sla_tier") != "Standard":
            raise ValueError("single-device topology drift")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 259200:
            raise ValueError("retry/duration drift")
        env = submit.get("env") or {}
        required = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-taskanchor-confirm-venv",
        }
        if any(env.get(key) != value for key, value in required.items()):
            raise ValueError("UAI/environment drift")
        shared = tuple(env.get(key) for key in (
            "MILESTONE_STEP", "MILESTONE_ADAPTER_SHA256",
            "MILESTONE_STATE_SHA256", "MILESTONE_RECEIPT_SHA256",
        ))
        if shared[0] not in {"200", "400", "600", "800", "1000"} or not all(valid_sha(x) for x in shared[1:]):
            raise ValueError("milestone binding drift")
        if common is None:
            common = shared
        elif shared != common:
            raise ValueError("cross-job milestone binding drift")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, str(job.get("name")))
        launcher = (
            "run_state8_taskanchor_confirmation_formal.sh"
            if job in formal else "run_state8_taskanchor_confirmation_c4.sh"
        )
        if f"bash scripts/{launcher}" not in command:
            raise ValueError("launcher drift")
    coordinates = {(job["submit_args"]["env"].get("SHARD_INDEX"), job["submit_args"]["env"].get("NUM_SHARDS")) for job in formal}
    if coordinates != {(str(index), "16") for index in range(16)}:
        raise ValueError("formal shard coordinate drift")
    seeds = {int(job["submit_args"]["env"].get("PAIR_SEED", -1)) for job in causal}
    if seeds != set(PAIR_HASHES):
        raise ValueError("C4 seed drift")
    for job in causal:
        env = job["submit_args"]["env"]
        if env.get("PAIR_MANIFEST_SHA256") != PAIR_HASHES[int(env["PAIR_SEED"])]:
            raise ValueError("C4 manifest hash drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    value = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid config")
    validate(value)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
