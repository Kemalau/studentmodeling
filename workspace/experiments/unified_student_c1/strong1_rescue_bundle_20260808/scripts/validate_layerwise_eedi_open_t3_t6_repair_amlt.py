#!/usr/bin/env python3
"""Fail-closed validation for the causal open-T3 T6-repair AMLT config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
SOURCE = "/mnt/amulet/studentmodeling/diagnostics/layerwise_task_retention_service_recovery/eedi_full_open_t3/state_path/lr2e-6_start12000_stop4250_97bef6748b1fa8d75809e9e01c9b76d57b6534d336b7037217289932a31df393_b926ee1acc6bf8a098693cef8705f88c739be214629dcea723daed6a0827c440_b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a/checkpoint-round-4250"
EXPECTED = {
    "layerwise-eedi-open-t3-t6-repair-balanced-lr5e7-seed42": (
        "5e-7",
        "manifests/layerwise_eedi_open_t3_t6_repair_balanced_seed42_v1.json",
        "08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9",
    ),
    "layerwise-eedi-open-t3-t6-repair-balanced-lr1e6-seed42": (
        "1e-6",
        "manifests/layerwise_eedi_open_t3_t6_repair_balanced_seed42_v1.json",
        "08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9",
    ),
    "layerwise-eedi-open-t3-t6-repair-strong-lr5e7-seed42": (
        "5e-7",
        "manifests/layerwise_eedi_open_t3_t6_repair_strong_seed42_v1.json",
        "5bb006010147bd99c0f1615ce0a54c016b8fdd15334b2f8b01a5c679f6443a09",
    ),
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("AMLT config is not a mapping")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(value: dict, selected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(value, sort_keys=False)
    require(not re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered), "unresolved placeholder")
    target = value.get("target") or {}
    require(target_matches(target, WORKSPACE), "target drift")
    environment = value.get("environment") or {}
    require(environment.get("image") == IMAGE, "image drift")
    require("msrcommonacr.azurecr.io/ml-common-py3.12:latest" not in rendered, "forbidden image")
    storage = (value.get("storage") or {}).get("student_store") or {}
    require(storage.get("storage_account_name") == "csnewsandfeeds4150361735", "storage drift")
    require(storage.get("container_name") == "amulet" and storage.get("mount_dir") == "/mnt/amulet", "mount drift")
    require(storage.get("is_output") is True, "storage output drift")
    code = value.get("code") or {}
    source_code_root = code.get("local_dir") == "$CONFIG_DIR/.."
    dumped_code_root = "local_dir" not in code and code.get("storage_id") == "default"
    require(source_code_root or dumped_code_root, "code root drift")
    require({".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])), "code ignore drift")
    jobs = {str(row.get("name")): row for row in value.get("jobs") or []}
    expected_names = selected_jobs or set(EXPECTED)
    require(set(jobs) == expected_names, "job set drift")
    warm_outputs: set[str] = set()
    for name, row in jobs.items():
        lr, manifest, digest = EXPECTED[name]
        require(row.get("sku") == "80G4-A100", f"SKU drift: {name}")
        require(row.get("process_count_per_node") == -1 and row.get("mpi") is True, f"MPI drift: {name}")
        require(str(row.get("sla_tier")).lower() == "standard", f"SLA drift: {name}")
        command = command_text(row.get("command"))
        require("run_layerwise_task_retention_rehearsal.sh" in command, f"command drift: {name}")
        validate_shell_syntax(command, name)
        env = (row.get("submit_args") or {}).get("env") or {}
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-layerwise-t6-repair-venv",
            "BRANCH": "eedi_full_open_t3",
            "LOSS_PROFILE": "state_path",
            "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_open_t3.py",
            "SEGMENT_ROWS": "1000",
            "LEARNING_RATE": lr,
            "EXPECTED_CAUSAL_PAIRS": "50",
            "DIAGNOSTIC_START_GLOBAL_POSITION": "17000",
            "DIAGNOSTIC_STOP_AFTER_ROUND": "4500",
            "MANIFEST": manifest,
            "MANIFEST_SHA256": digest,
            "WARM_START_SOURCE_CHECKPOINT_DIR": SOURCE,
            "WARM_START_PRIOR_MANIFEST": "manifests/layerwise_task_retention_eedi_full_seed42_v1.json",
            "WARM_START_EXPECTED_ROUND": "4250",
            "WARM_START_EXPECTED_WORLD_SIZE": "4",
        }
        for key, expected_value in expected.items():
            require(str(env.get(key)) == expected_value, f"{key} drift: {name}")
        warm = str(env.get("WARM_START_CHECKPOINT_DIR", ""))
        require(warm.startswith("/tmp/") and warm not in warm_outputs, f"warm output drift: {name}")
        warm_outputs.add(warm)
        require(int((row.get("submit_args") or {}).get("max_attempts", 0)) == 2, f"retry drift: {name}")
        require(int((row.get("submit_args") or {}).get("max_run_duration_seconds", 0)) == 172800, f"duration drift: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    selected = set(args.jobs or ()) or None
    unknown = (selected or set()).difference(EXPECTED)
    if unknown:
        raise ValueError(f"Unknown selected jobs: {sorted(unknown)}")
    validate(load(args.config), selected)
    print(json.dumps({"validated": str(args.config), "jobs": sorted(selected or EXPECTED)}))


if __name__ == "__main__":
    main()
