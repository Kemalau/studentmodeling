#!/usr/bin/env python3
"""Fail-closed validation for a materialized strong1 rescue config."""

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
    from materialize_state8_strong1_rescue_amlt import (
        ARMS,
        IMAGE,
        WORKSPACE,
        load_selection,
    )
except ImportError:
    from scripts.materialize_state8_strong1_rescue_amlt import (
        ARMS,
        IMAGE,
        WORKSPACE,
        load_selection,
    )


HASHES = {
    "scripts/run_state8_strong1_rescue.sh": "3311d038488b0aef54ac59798bab8b2e21b58dd9fcd2ce3495a067332314b641",
    "scripts/run_state8_stateonly_repair_ab.sh": "2d52d2d84b64dcc1d177583ddb96b006e6afdeee74cb2d3241325ae170a059b7",
    "scripts/train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "scripts/train_state8_task_anchored_causal_repair.py": "b622a62fffef45a19b3b02b771cee4412d5f9d17f1a3fe99c6ba046e6111e66b",
    "scripts/train_state8_taxonomy_causal_repair.py": "26e62969a03aab89047dd42d5bbe6d144eae9bb77d0fa0b962b0da89b9b75347",
    "scripts/train_state8_diagnosis_first_causal_repair.py": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
    "scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py": "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py": "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b",
    "scripts/evaluate_recurrent_causal_gate.py": "e0c9fe1361dfda5eddc14abcd275fbe8e2b92f472f9eaf27468b71f800fafccf",
    "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    config_path: Path,
    selection_path: Path,
    bundle: Path,
    expected_jobs: set[str] | None = None,
) -> None:
    selected = load_selection(selection_path)["selected"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True,
    }:
        raise ValueError("Storage drift")
    if (config.get("code") or {}).get("local_dir") != "$CONFIG_DIR/../strong1_rescue_bundle_20260808":
        raise ValueError("Code root drift")
    jobs = config.get("jobs") or []
    expected_names = {
        f"state8-strong1-rescue-{arm}-seed{settings['seed']}"
        for arm, settings in ARMS.items()
    }
    requested = expected_jobs or expected_names
    if not requested or not requested.issubset(expected_names):
        raise ValueError("Unknown requested rescue job")
    if {str(job.get("name")) for job in jobs} != requested:
        raise ValueError("Job set drift")
    for job in jobs:
        name = str(job["name"])
        arm = next(key for key in ARMS if f"-{key}-" in name)
        settings = ARMS[arm]
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G1-A100", 1, False, "Standard"
        ):
            raise ValueError(f"Topology drift: {name}")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 345600:
            raise ValueError(f"Retry drift: {name}")
        expected = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "STUDENT_ROOT": selected["remote_root"],
            "EXPECTED_STUDENT_ADAPTER_SHA256": selected["artifact_sha256"]["adapter_model"],
            "EXPECTED_STUDENT_STATE_SHA256": selected["artifact_sha256"]["recurrent_state"],
            "STRONG1_SOURCE_SLUG": selected["slug"],
            "REQUESTED_WORLD_SIZE": "1", "GRADIENT_ACCUMULATION_STEPS": "4",
            "STATE8_REPAIR_SAVE_STEPS": "500",
            "STATE8_REPAIR_RECOVERY_SAVE_STEPS": "500",
            "REPAIR_ARM": settings["repair_arm"], "TRAIN_ENTRYPOINT": settings["entrypoint"],
            "TRAIN_SEED": settings["seed"], "RUN_TAG": settings["tag"],
            "STATE8_CAUSAL_REPEAT_FRACTION": settings["repeat"],
            "STATE8_CAUSAL_WEIGHT": settings["causal_weight"],
            "STATE8_CAUSAL_LABEL_WEIGHT": settings["label_weight"],
            "STATE8_REPAIR_LABEL_WEIGHT": settings["label_weight"],
            "STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256": HASHES["scripts/train_state8_diagnosis_first_causal_repair.py"],
            "STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256": HASHES["scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py"],
            "STATE8_DIAGNOSIS_FIRST_C4_SHA256": HASHES["scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py"],
        }
        observed = {key: str(value) for key, value in env.items()}
        for key, value in expected.items():
            if observed.get(key) != str(value):
                raise ValueError(f"Environment drift: {name}/{key}")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        if "run_state8_strong1_rescue.sh" not in command:
            raise ValueError(f"Entrypoint drift: {name}")
    for relative, expected in HASHES.items():
        observed = digest(bundle / relative)
        if observed != expected:
            raise ValueError(f"Bundle hash drift: {relative}: {observed}")
    launcher = bundle / "scripts/run_state8_strong1_rescue.sh"
    if not launcher.is_file():
        raise ValueError("Strong1 rescue wrapper is missing")
    source = launcher.read_text(encoding="utf-8")
    for fragment in (
        "normalized_from_stage", "EXPECTED_STUDENT_ADAPTER_SHA256",
        "EXPECTED_STUDENT_STATE_SHA256", "--num-pairs 150",
        "evaluate_recurrent_causal_gate.py", "strong1_source_immutable_receipt.json",
        "train_state8_diagnosis_first_causal_repair.py",
        "STATE8_REPAIR_LABEL_WEIGHT",
    ):
        if fragment not in source:
            raise ValueError(f"Rescue wrapper drift: {fragment}")
    repair_launcher = (bundle / "scripts/run_state8_stateonly_repair_ab.sh").read_text(
        encoding="utf-8"
    )
    for fragment in (
        "STATE8_REPAIR_SAVE_STEPS % GRADIENT_ACCUMULATION_STEPS",
        "STATE8_REPAIR_RECOVERY_SAVE_STEPS % GRADIENT_ACCUMULATION_STEPS",
    ):
        if fragment not in repair_launcher:
            raise ValueError(f"Rescue checkpoint boundary guard drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*")
    args = parser.parse_args()
    validate(
        args.config,
        args.selection,
        args.bundle,
        set(args.jobs) if args.jobs else None,
    )
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
