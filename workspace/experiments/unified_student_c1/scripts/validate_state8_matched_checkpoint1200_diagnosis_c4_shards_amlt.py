#!/usr/bin/env python3
"""Fail-closed AMLT validation for matched checkpoint-1200 C4 shards."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
EXPECTED = {f"state8-matched-cp1200-diagnosis-c4-shard{start:02d}": start for start in (0, 10, 20, 30, 40)}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    return value


def validate(
    config: dict,
    *,
    source_root: Path,
    expected_names: set[str] | None = None,
) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store")
    if storage != {"storage_account_name": "csnewsandfeeds4150361735", "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True}:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (code.get("local_dir") == "$CONFIG_DIR/.." or (code.get("local_dir") is None and code.get("storage_id") == "default")):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusion drift")
    jobs = config.get("jobs") or []
    expected_names = set(EXPECTED) if expected_names is None else expected_names
    if not expected_names or not expected_names.issubset(EXPECTED):
        raise ValueError("Invalid expected job subset")
    if {str(job.get("name")) for job in jobs} != expected_names:
        raise ValueError("Shard jobs drift")
    starts = []
    for job in jobs:
        name = str(job["name"])
        if job.get("sku") != "80G1-A100" or job.get("process_count_per_node") != 1 or job.get("mpi") is not False or job.get("sla_tier") != "Standard":
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        start = int(env.get("PAIR_START", -1)); starts.append(start)
        if start != EXPECTED[name] or int(env.get("PAIR_COUNT", -1)) != 10:
            raise ValueError(f"Pair range drift: {name}")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry/duration drift: {name}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if "run_state8_matched_checkpoint1200_diagnosis_c4_shard.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    if sorted(starts) != sorted(EXPECTED[name] for name in expected_names):
        raise ValueError("Non-disjoint pair coverage")
    launcher = (source_root / "scripts/run_state8_matched_checkpoint1200_diagnosis_c4_shard.sh").read_text(encoding="utf-8")
    launcher_sha256 = hashlib.sha256(launcher.encode()).hexdigest()
    if rendered.count(launcher_sha256) != len(expected_names):
        raise ValueError("Launcher SHA256 binding drift")
    for fragment in ("checkpoint-1200", "--num-pairs 50", '--pair-start "$PAIR_START"', '--pair-count "$PAIR_COUNT"', "27fcdc000fcf0d410004aad84f78b314fc9bc7d227385aa62c67f9eabeedc356", "7a99e2f15379549d3caef6f0d764b97716e143c4881e5dec4e2a54caf184d12e", "evaluate_recurrent_diagnosis_c4_diagnosis_first.py"):
        if fragment not in launcher:
            raise ValueError(f"Launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", action="append", choices=sorted(EXPECTED))
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    validate(
        load(args.config),
        source_root=root,
        expected_names=(set(args.job) if args.job else None),
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
