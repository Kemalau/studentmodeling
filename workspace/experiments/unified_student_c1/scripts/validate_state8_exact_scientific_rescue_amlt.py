#!/usr/bin/env python3
"""Fail-closed validator for the conditional exact State8 rescue config."""

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

try:
    from materialize_state8_exact_scientific_rescue_amlt import (
        JOB_BY_MODE,
        load_failed_selection,
        required_modes,
    )
except ImportError:
    from scripts.materialize_state8_exact_scientific_rescue_amlt import (
        JOB_BY_MODE,
        load_failed_selection,
        required_modes,
    )


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
UAI = "rankfun_aml"
SOURCE_ROOT = (
    "/mnt/amulet/studentmodeling/diagnostics/"
    "state8_stateonly_repair_seed65_trajectory01_"
    "exact_teacher_curriculum_c4_seed65_v1"
)
HASHES = {
    "scripts/run_state8_exact_scientific_rescue.sh": "5fa5ce56e7215342b79a5321cae83a3bf440a97bbd0ed77d3956e573390cece6",
    "scripts/run_state8_stateonly_repair_ab.sh": "2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7",
    "scripts/train_recurrent_student_m4.py": "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
    "scripts/train_state8_diagnosis_first_causal_repair.py": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
    "scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py": "eb9837127af9fe3761f6fa57b78a3e9ba3373a223706f0a4921749eab61e35a3",
    "scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py": "c8e4e2bebf7ebf01b15685b74a5bdcbaa0afcb5a16c774b40ba0d7070819095a",
    "scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py": "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py": "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b",
    "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
    "manifests/layerwise_eedi_open_t3_t6_repair_balanced_seed42_v1.json": "08945e930bc8b9a55e57a41d4656612e3dfce99b8544c88068110a74a1a22da9",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(config_path: Path, selection_path: Path, bundle: Path) -> None:
    selection = load_failed_selection(selection_path)
    candidate = selection["candidates"][0]
    modes = required_modes(candidate)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED(?:_AT_CONDITIONAL_MATERIALIZATION)?\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("code") or {}).get("local_dir") != (
        "$CONFIG_DIR/../state8_exact_scientific_rescue_bundle_20260808"
    ):
        raise ValueError("Bundle root drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if {str(job.get("name")) for job in jobs} != {JOB_BY_MODE[mode] for mode in modes}:
        raise ValueError("Conditional job set drift")
    for job in jobs:
        if (
            job.get("sku"), job.get("process_count_per_node"),
            job.get("mpi"), job.get("sla_tier")
        ) != ("80G1-A100", 1, False, "Standard"):
            raise ValueError(f"Topology drift: {job.get('name')}")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        mode = str(env.get("STATE8_RESCUE_MODE") or "")
        if mode not in modes or job.get("name") != JOB_BY_MODE[mode]:
            raise ValueError("Mode/name drift")
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 86400:
            raise ValueError("Retry policy drift")
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": UAI,
            "EXACT_SOURCE_ROOT": SOURCE_ROOT,
            "EXPECTED_STUDENT_ADAPTER_SHA256": candidate["artifact_sha256"]["adapter_model"],
            "EXPECTED_STUDENT_STATE_SHA256": candidate["artifact_sha256"]["recurrent_state"],
            "STATE8_COMPOSED_ENTRYPOINT_SHA256": HASHES[
                "scripts/train_state8_diagnosis_first_task_anchored_causal_repair.py"
            ],
            "STATE8_RETENTION_ENTRYPOINT_SHA256": HASHES[
                "scripts/train_state8_diagnosis_first_retention_anchored_causal_repair.py"
            ],
            "STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256": HASHES[
                "scripts/train_state8_diagnosis_first_causal_repair.py"
            ],
        }
        for key, value in expected.items():
            if str(env.get(key)) != value:
                raise ValueError(f"Environment drift: {job.get('name')}/{key}")
        command = command_text(job.get("command")); validate_shell_syntax(command, str(job.get("name")))
        if "run_state8_exact_scientific_rescue.sh" not in command:
            raise ValueError("Entrypoint drift")
    for relative, expected in HASHES.items():
        observed = digest(bundle / relative)
        if observed != expected:
            raise ValueError(f"Bundle hash drift: {relative}: {observed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config, args.selection, args.bundle)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
