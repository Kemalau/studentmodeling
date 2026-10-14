#!/usr/bin/env python3
"""Fail-closed preflight for the two immutable consolidation probes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

try:
    from amlt_validation_utils import command_text, target_matches, validate_shell_syntax
except ImportError:
    from scripts.amlt_validation_utils import (
        command_text,
        target_matches,
        validate_shell_syntax,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "state8-taxonomy-consolidation-t3heavy-r100-probe": "t3heavy",
    "state8-taxonomy-consolidation-t3heavy-diag-r100-probe": "t3heavy_diag",
}


def validate(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False).replace("<|vision_pad|>", "")
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
    code = config.get("code") or {}
    if not (
        code.get("local_dir") == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusion drift")
    jobs = config.get("jobs") or []
    job_names = {job.get("name") for job in jobs}
    if not job_names or not job_names.issubset(JOBS) or len(jobs) != len(job_names):
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 21600:
            raise ValueError(f"Retry drift: {name}")
        if (submit.get("env") or {}) != {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-consolidation-probe-venv",
            "PROBE_VARIANT": JOBS[name],
            "PROBE_ROUND": "100",
        }:
            raise ValueError(f"Environment drift: {name}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        for fragment in (
            "set -euo pipefail",
            "FAILED line=%s rc=%s command=%s",
            "bash scripts/run_state8_taxonomy_consolidation_milestone_probe.sh",
        ):
            if fragment not in command:
                raise ValueError(f"Launcher drift: {fragment}")
    launcher = (Path(__file__).parent / "run_state8_taxonomy_consolidation_milestone_probe.sh").read_text()
    for fragment in (
        "state8_taxonomy_consolidation_milestones_20260808",
        "receipt['world_size']==4",
        "--tasks diagnose_misconception",
        "--sources eedi_mining malrule",
        "evaluate_recurrent_diagnosis_c4_taxonomy.py",
        "--num-pairs 10",
        "raw_history_visible_to_decoder",
        "persistent_history_kv_cache",
        "['native','noop','samemal','random','swap']",
    ):
        if fragment not in launcher:
            raise ValueError(f"Probe contract drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
