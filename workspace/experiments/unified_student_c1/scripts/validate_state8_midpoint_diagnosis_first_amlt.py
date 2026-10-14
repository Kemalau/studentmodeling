#!/usr/bin/env python3
"""Fail-closed preflight for the step-2000 diagnosis-first repair screen."""

from __future__ import annotations

import argparse
import hashlib
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
JOB = "state8-midpoint-diagnosis-first-seed63"
CHECKPOINT = (
    "/mnt/amulet/studentmodeling/checkpoints/"
    "state8_qualified_teacher_curriculum_seed42_exact5000_rounds1250_kd1_v2_20260805_"
    "taxonomy_semantic_v1_strong1_widen8_single_lr5e7_seed60_single_ga4/checkpoint-2000"
)
COMMON = {
    "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
    "VENV_ROOT": "/tmp/studentmodeling-state8-midpoint-diagnosis-first-venv",
    "STATE8_CHECKPOINT_ROOT": CHECKPOINT,
    "EXPECTED_STUDENT_ADAPTER_SHA256": "af09ff4b886fefe42b99d60edf467f5c3c7830b8caaa909668b189b12d9ae6f5",
    "EXPECTED_STUDENT_STATE_SHA256": "2bcd1cf600b699010e2d7de8bd8fe3b310fa500c38ba66080fa1f5a54b1ede7f",
    "REPAIR_ARM": "trajectory01",
    "REQUESTED_WORLD_SIZE": "1",
    "GRADIENT_ACCUMULATION_STEPS": "4",
    "STATE8_REPAIR_SAVE_STEPS": "200",
    "STATE8_REPAIR_RECOVERY_SAVE_STEPS": "200",
    "TRAIN_ENTRYPOINT": "scripts/train_state8_diagnosis_first_causal_repair.py",
    "RUN_TAG": "midpoint_step2000_diagnosis_first_v1",
    "TRAIN_SEED": "63",
    "STATE8_REPAIR_LABEL_WEIGHT": "1.0",
    "STATE8_CAUSAL_LABEL_WEIGHT": "1.0",
    "STATE8_RANDOM_NEUTRAL": "1",
    "STATE8_GEOMETRY_WEIGHT": "0.10",
    "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
    "STATE8_DIFFERENT_MAX_COSINE": "0.90",
    "STATE8_CAUSAL_DIAGNOSIS_ONLY": "0",
    "STATE8_CAUSAL_REPEAT_FRACTION": "0.50",
    "STATE8_CAUSAL_WEIGHT": "2.0",
    "STATE8_CAUSAL_MARGIN": "0.5",
    "STATE8_LAYERWISE_TRANSITION_ONLY": "0",
    "STATE8_UPDATER_ONLY": "0",
    "STATE8_FORMAL_SCOPE": "unified",
    "STATE8_REPAIR_MANIFEST": "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json",
    "STATE8_REPAIR_MANIFEST_SHA256": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
    "STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
    "STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256": "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
    "STATE8_DIAGNOSIS_FIRST_C4_SHA256": "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b",
}
HASHES = {
    "scripts/run_state8_midpoint_diagnosis_first_repair.sh": "376998a3304981dfb2abfd84f3ed1d0d32ff24137487c7c85377a5e25e702b64",
    "scripts/run_state8_stateonly_repair_ab.sh": "de009449b6d9ecb944094d19dca3b673de497b04fd1ce7ff0f3582046fa129b2",
    "scripts/select_latest_valid_single_checkpoint.py": "224736e342bb55bd54ad284eb0907a880dccf9bb906e4d22d71b91edb618d243",
    "scripts/train_recurrent_student_m4.py": "ae502968872ff6597e73bcc12cb3ff9a5bbb0fe9d7a763383f6e76bc392de42c",
    "scripts/train_state8_diagnosis_first_causal_repair.py": COMMON["STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256"],
    "scripts/evaluate_recurrent_formal_protocols_diagnosis_first.py": COMMON["STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256"],
    "scripts/evaluate_recurrent_diagnosis_c4_diagnosis_first.py": COMMON["STATE8_DIAGNOSIS_FIRST_C4_SHA256"],
    "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json": COMMON["STATE8_REPAIR_MANIFEST_SHA256"],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path, bundle: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<[^>]+>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    code = config.get("code") or {}
    code_root = code.get("local_dir")
    if code_root is not None:
        if code_root != "$CONFIG_DIR/../state8_midpoint_diagnosis_first_bundle_20260808":
            raise ValueError("Code root drift")
    elif code.get("storage_id") != "default":
        # AMLT --dump replaces the already validated local upload root with
        # the resolved project storage binding.
        raise ValueError("Rendered code storage drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet", "mount_dir": "/mnt/amulet", "is_output": True,
    }:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
        "80G1-A100", 1, False, "Standard"
    ):
        raise ValueError("Topology drift")
    submit = job.get("submit_args") or {}
    if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != 43200:
        raise ValueError("Retry/duration drift")
    observed = {key: str(value) for key, value in (submit.get("env") or {}).items()}
    if observed != COMMON:
        raise ValueError("Environment drift")
    command = command_text(job.get("command")); validate_shell_syntax(command, JOB)
    if "run_state8_midpoint_diagnosis_first_repair.sh" not in command:
        raise ValueError("Entrypoint drift")
    for relative, expected in HASHES.items():
        actual = digest(bundle / relative)
        if actual != expected:
            raise ValueError(f"Bundle hash drift: {relative}: {actual}")
    source = (bundle / "scripts/run_state8_midpoint_diagnosis_first_repair.sh").read_text()
    for fragment in (
        "initialization_only_not_complete_curriculum", "history_dropout_counts",
        "midpoint-diagnosis-first", "--num-pairs 150", "answer_c4_150",
    ):
        if fragment not in source:
            raise ValueError(f"Source contract drift: {fragment}")
    launcher = (bundle / "scripts/run_state8_stateonly_repair_ab.sh").read_text()
    if "select_latest_valid_single_checkpoint.py" not in launcher:
        raise ValueError("Single-device resume is not enabled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config, args.bundle)
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
