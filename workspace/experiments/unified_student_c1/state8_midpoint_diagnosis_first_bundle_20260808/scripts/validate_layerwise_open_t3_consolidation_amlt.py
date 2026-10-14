#!/usr/bin/env python3
"""Fail-closed validator for the two-arm phase-2 consolidation experiment."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
MANIFEST = "manifests/layerwise_open_t3_consolidation_equal_t3_t6_seed42_v1.json"
MANIFEST_SHA256 = "3a6afeecb7949f72f464efcf608402b3a258e88609f4d8d45bf0bbd252a26b16"
SOURCE = "/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/equal_t3_t6_open_t3/state_kd/lr5e-6_start12000_stop3250_32adfcc7bd1d78f541ec8685ec8f96f9934b35471011bb896c402b1e77a7bf6f_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_2474b77459701fa36ec564c84b7f3b6b0d158c9ab128b5147111d252662b0931/checkpoint-round-3250"
SPECS = {
    "layerwise-open-t3-consolidation-statekd-lr5e7-seed42": (
        "state_kd",
        "scripts/train_recurrent_student_m4_open_t3_selective_kd.py",
        "/tmp/layerwise-open-t3-consolidation-statekd-warm-start",
    ),
    "layerwise-open-t3-consolidation-statepath-lr5e7-seed42": (
        "state_path",
        "scripts/train_recurrent_student_m4_open_t3.py",
        "/tmp/layerwise-open-t3-consolidation-statepath-warm-start",
    ),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT YAML")
    return value


def validate(config: dict, expected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
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
    code = config.get("code") or {}
    # ``amlt run --dump`` resolves ``local_dir`` into the project's default
    # storage upload and therefore emits ``storage_id: default`` instead.  The
    # recovery controller validates both the source config and that normalized
    # dump before it submits.  Accept exactly those two representations; this
    # keeps the code-root gate fail-closed without rejecting AMLT's own dump.
    source_code_root = code.get("local_dir") == "$CONFIG_DIR/.."
    dumped_code_root = (
        "local_dir" not in code and code.get("storage_id") == "default"
    )
    if not (source_code_root or dumped_code_root):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code ignore drift")
    expected = set(SPECS) if expected_jobs is None else set(expected_jobs)
    if not expected or not expected.issubset(SPECS):
        raise ValueError("Expected job set drift")
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != expected or len(jobs) != len(expected):
        raise ValueError("Job set drift")
    commands = []
    for job in jobs:
        name = str(job["name"])
        profile, entrypoint, warm_dir = SPECS[name]
        if (
            job.get("sku") != "80G4-A100"
            or job.get("process_count_per_node") != -1
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"Retry/duration drift: {name}")
        env = submit.get("env") or {}
        required = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-layerwise-open-t3-consolidation-venv",
            "BRANCH": "equal_t3_t6_open_t3",
            "LOSS_PROFILE": profile,
            "TRAIN_ENTRYPOINT": entrypoint,
            "SEGMENT_ROWS": "1000",
            "LEARNING_RATE": "5e-7",
            "EXPECTED_CAUSAL_PAIRS": "50",
            "DIAGNOSTIC_START_GLOBAL_POSITION": "13000",
            "DIAGNOSTIC_STOP_AFTER_ROUND": "3500",
            "MANIFEST": MANIFEST,
            "MANIFEST_SHA256": MANIFEST_SHA256,
            "WARM_START_SOURCE_CHECKPOINT_DIR": SOURCE,
            "WARM_START_CHECKPOINT_DIR": warm_dir,
            "WARM_START_PRIOR_MANIFEST": "manifests/layerwise_task_retention_equal_t3_t6_seed42_v1.json",
            "WARM_START_EXPECTED_ROUND": "3250",
            "WARM_START_EXPECTED_WORLD_SIZE": "4",
        }
        observed = {key: str(env.get(key)) for key in required}
        if observed != required:
            raise ValueError(f"Scientific env drift: {name}: {observed}")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.append(command.strip())
    if set(commands) != {"set -euo pipefail\nbash scripts/run_layerwise_task_retention_rehearsal.sh"}:
        raise ValueError("Launcher drift")

    root = Path(__file__).resolve().parent
    launcher = (root / "run_layerwise_task_retention_rehearsal.sh").read_text(encoding="utf-8")
    for fragment in (
        "prepare_layerwise_consolidation_warm_start.py",
        "audit_layerwise_diagnostic_warm_start.py",
        '--diagnostic-warm-start-checkpoint "$CHECKPOINT/recurrent_state.pt"',
        "--causal-objective-mode specificity_triplet",
        "--normalize-causal-exposure",
        "--save-steps 50 --checkpoint-retention 2",
    ):
        if fragment not in launcher:
            raise ValueError(f"Launcher invariant missing: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(load(args.config), set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
