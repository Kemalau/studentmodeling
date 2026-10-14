#!/usr/bin/env python3
"""Fail-closed validation for the exposure-matched C3 Feeds jobs."""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ARMS = {
    "c3-matched-malrule-correctness-seed20260804": (
        "malrule_correctness",
        "manifests/c3_exposure_matched_20260804/c3_exposure_matched_malrule_correctness_seed20260804_v1.json",
        "ce358ed02a74fe0d5b9bdf75073e9202eff0fe689a5620fb10f8b02e0e632d25",
    ),
    "c3-matched-malrule-response-seed20260804": (
        "malrule_response",
        "manifests/c3_exposure_matched_20260804/c3_exposure_matched_malrule_response_seed20260804_v1.json",
        "e6202050ea4dbc7b00298546abb7d6e964565a807fbfda8c0165f9ca4c8dd6ea",
    ),
    "c3-matched-malrule-misconception-seed20260804": (
        "malrule_misconception",
        "manifests/c3_exposure_matched_20260804/c3_exposure_matched_malrule_misconception_seed20260804_v1.json",
        "ba74efc1eb675cab148af062fde1c6bad0c9b57fbbc1c579b68756ef340da818",
    ),
}
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-c3-matched-venv",
    "BRANCH": "eedi_full",
    "LOSS_PROFILE": "state_path",
    "TRAIN_ENTRYPOINT": "scripts/train_recurrent_student_m4_open_t3.py",
    "CAUSAL_VARIANT": "base",
    "SAME_RULE_AUGMENTER_SHA256": "d0f5eb225e771a745874ab9a6c829b8cd4ac2bd71754ae96f8c8007498bc1ff2",
    "SEGMENT_ROWS": "1000",
    "LEARNING_RATE": "5e-7",
    "EXPECTED_CAUSAL_PAIRS": "50",
    "DIAGNOSTIC_START_GLOBAL_POSITION": "24000",
    "DIAGNOSTIC_STOP_AFTER_ROUND": "6250",
    "WARM_START_CHECKPOINT_DIR": "/mnt/amulet/studentmodeling/checkpoints/layerwise_unified_world_model_seed42_taskretention_eedi_full_open_t3_state_path_v1_20260802/checkpoint-round-1500",
    "WARM_START_EXPECTED_ROUND": "1500",
    "WARM_START_EXPECTED_WORLD_SIZE": "16",
    "C3_CONTROLLED_EXPERIMENT": "exposure_matched_auxiliary_supervision_v1",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    return value


def validate(config: dict, *, selected_jobs: set[str] | None = None) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|<describe your experiment>", rendered):
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
    expected = set(ARMS) if selected_jobs is None else set(selected_jobs)
    if not expected or not expected <= set(ARMS):
        raise ValueError("Unknown selected C3 job")
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != expected:
        raise ValueError("C3 job-set drift")
    normalized = {}
    commands = set()
    for job in jobs:
        name = str(job["name"])
        if (
            job.get("sku") != "80G4-A100"
            or job.get("mpi") is not True
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError(f"C3 GPU/MPI/SLA drift: {name}")
        submit = job.get("submit_args") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 172800:
            raise ValueError(f"C3 retry/duration drift: {name}")
        arm, manifest, digest = ARMS[name]
        env = {str(k): str(v) for k, v in (submit.get("env") or {}).items()}
        expected_env = {
            **COMMON,
            "C3_ARM": arm,
            "MANIFEST": manifest,
            "MANIFEST_SHA256": digest,
        }
        if env != expected_env:
            raise ValueError(f"C3 controlled environment drift: {name}")
        common = copy.deepcopy(env)
        for key in ("C3_ARM", "MANIFEST", "MANIFEST_SHA256"):
            common.pop(key)
        normalized[name] = common
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        commands.add(command)
    if len(commands) != 1 or len({tuple(sorted(row.items())) for row in normalized.values()}) != 1:
        raise ValueError("C3 arms differ beyond auxiliary supervision")
    command = next(iter(commands)).replace("$$", "$")
    for fragment in (
        "a6e9ac56f263caa1982403979a6b4ff70750ea2f95086cdd6087edeca4488bc8",
        "FAILED line=%s rc=%s command=%s",
        "b5e7a5c301280df64083dadf20273902747bcf8ea4e4cdd03de3e49d4e53187a",
        'sha256sum "$MANIFEST"',
        "run_layerwise_task_retention_rehearsal.sh",
    ):
        if fragment not in command:
            raise ValueError(f"C3 provenance command drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", choices=tuple(ARMS))
    args = parser.parse_args()
    validate(load(args.config), selected_jobs=set(args.jobs) if args.jobs else None)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
