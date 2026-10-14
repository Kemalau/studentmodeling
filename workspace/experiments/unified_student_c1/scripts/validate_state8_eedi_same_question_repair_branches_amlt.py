#!/usr/bin/env python3
"""Fail-closed validation for bound State8 repair-branch AMLT configs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


APPROVED_IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts/run_state8_eedi_same_question_repair_branch.sh"
REQUIRED_ENV = {
    "_AZUREML_SINGULARITY_JOB_UAI", "VENV_ROOT", "REPAIR_BRANCH", "REPAIR_SEED",
    "SOURCE_CHECKPOINT_ARM", "SOURCE_CHECKPOINT_STEP", "SOURCE_CHECKPOINT_ROOT",
    "SOURCE_ANCHOR_MODE", "EXPECTED_STUDENT_ADAPTER_SHA256",
    "EXPECTED_STUDENT_STATE_SHA256", "STATE8_EEDI_CAUSAL_TWINS",
    "REPAIR_REMOTE_ROOT",
    "STATE8_CAUSAL_REPEAT_FRACTION", "STATE8_CAUSAL_WEIGHT",
    "STATE8_CAUSAL_MARGIN", "STATE8_EEDI_GEOMETRY_WEIGHT",
}
OPTIONAL_ENV = {
    "STATE8_CONTROL_RANDOM_SAMPLES",
    "STATE8_CONTROL_RANDOM_WEIGHT",
    "STATE8_CONTROL_SAME_RULE_WEIGHT",
    "STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT",
    "STATE8_CAUSAL_TEACHER_COSINE_WEIGHT",
    "STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT",
    "STATE8_EARLY_LABEL_BINDING_WEIGHT",
}


def validate(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    if any(marker in raw for marker in ("REQUIRED", "PENDING", "HASH_PENDING", "<...>")):
        raise ValueError("Unresolved placeholder in repair config")
    config = yaml.safe_load(raw)
    target = config.get("target") or {}
    template_target = {
        "service": "manifold",
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity",
    }
    dumped_target = {
        "name": "Feeds",
        "workspace_name": "CS-NewsAndFeeds-Singularity",
        "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
    }
    dumped_sing_target = {"service": "sing", **dumped_target}
    if target not in (template_target, dumped_target, dumped_sing_target):
        raise ValueError(f"Target drift: {target}")
    if (config.get("environment") or {}).get("image") != APPROVED_IMAGE:
        raise ValueError("Image drift")
    storage = (config.get("storage") or {}).get("student_store") or {}
    if storage != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError(f"Storage drift: {storage}")
    jobs = config.get("jobs") or []
    if not 1 <= len(jobs) <= 3:
        raise ValueError("Repair job count must be one to three")
    names: set[str] = set()
    runner_sha256 = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    for job in jobs:
        name = str(job.get("name") or "")
        if name in names or not name.startswith("state8-eedi-repair-"):
            raise ValueError(f"Invalid repair job name: {name}")
        names.add(name)
        if job.get("sku") != "80G1-A100" or int(job.get("process_count_per_node", 0)) != 1:
            raise ValueError(f"GPU topology drift: {name}")
        if job.get("sla_tier") != "Standard" or job.get("mpi") is not False:
            raise ValueError(f"SLA/launcher drift: {name}")
        command = str(job.get("command") or "")
        if "run_state8_eedi_same_question_repair_branch.sh" not in command:
            raise ValueError(f"Runner drift: {name}")
        if runner_sha256 not in command:
            raise ValueError(f"Runner binding drift: {name}")
        env = ((job.get("submit_args") or {}).get("env") or {})
        if not REQUIRED_ENV <= set(env) <= REQUIRED_ENV | OPTIONAL_ENV:
            raise ValueError(f"Environment schema drift: {name}")
        if env["_AZUREML_SINGULARITY_JOB_UAI"] != "rankfun_aml":
            raise ValueError(f"UAI drift: {name}")
        root = str(env["SOURCE_CHECKPOINT_ROOT"])
        source_arm = str(env["SOURCE_CHECKPOINT_ARM"])
        source_step = str(env["SOURCE_CHECKPOINT_STEP"])
        if source_arm in {"updater", "full"}:
            if source_step not in {"20", "100"}:
                raise ValueError(f"Legacy source step drift: {name}")
            expected = (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                f"state8_eedi_same_question_20260811/{source_arm}/checkpoint-{source_step}"
            )
        elif source_arm in {"curriculum_generic", "curriculum_current_query"}:
            if source_step not in {"2000", "3000", "4000", "5000"}:
                raise ValueError(f"Curriculum source step drift: {name}")
            tag = (
                "state8_diagnosis_first_eedi_curriculum_20260812"
                if source_arm == "curriculum_generic"
                else "state8_diagnosis_first_eedi_current_query_curriculum_20260812"
            )
            expected = (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                f"{tag}/checkpoint-{source_step}"
            )
        else:
            raise ValueError(f"Source arm drift: {name}")
        if root != expected:
            raise ValueError(f"Source checkpoint drift: {name}")
        branch = str(env["REPAIR_BRANCH"])
        seed = str(env["REPAIR_SEED"])
        mode = str(env["SOURCE_ANCHOR_MODE"])
        anchor = "updater" if mode == "updater_mse1" else "full"
        expected_output = (
            "/mnt/amulet/studentmodeling/diagnostics/"
            f"state8_stateonly_repair_seed{seed}_taskanchor_source_anchor_"
            f"{anchor}_mse1_eedi_same_question_v1_{branch}_seed{seed}"
        )
        if env["REPAIR_REMOTE_ROOT"] != expected_output:
            raise ValueError(f"Repair output root drift: {name}")
        for key in ("EXPECTED_STUDENT_ADAPTER_SHA256", "EXPECTED_STUDENT_STATE_SHA256"):
            value = str(env[key])
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Invalid {key}: {name}")
        if "STATE8_CONTROL_RANDOM_SAMPLES" in env:
            if int(env["STATE8_CONTROL_RANDOM_SAMPLES"]) < 2:
                raise ValueError(f"Too few random controls: {name}")
            if float(env.get("STATE8_CONTROL_RANDOM_WEIGHT", 0)) <= 0:
                raise ValueError(f"Random-control weight is not positive: {name}")
            if float(env.get("STATE8_CONTROL_SAME_RULE_WEIGHT", 0)) <= 0:
                raise ValueError(f"Same-rule-control weight is not positive: {name}")
        teacher_keys = {
            "STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT",
            "STATE8_CAUSAL_TEACHER_COSINE_WEIGHT",
            "STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT",
        }
        present_teacher_keys = teacher_keys & set(env)
        if present_teacher_keys and present_teacher_keys != teacher_keys:
            raise ValueError(f"Incomplete causal-teacher binding: {name}")
        if present_teacher_keys:
            if not (
                str(env["REPAIR_BRANCH"]).startswith("causal_teacher_")
                or str(env["REPAIR_BRANCH"]).startswith("early_label_")
            ):
                raise ValueError(f"Causal-teacher override on wrong branch: {name}")
            if float(env["STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT"]) <= 0:
                raise ValueError(f"Causal-teacher weight is not positive: {name}")
            components = (
                float(env["STATE8_CAUSAL_TEACHER_COSINE_WEIGHT"]),
                float(env["STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT"]),
            )
            if min(components) < 0 or sum(components) <= 0:
                raise ValueError(f"Invalid causal-teacher components: {name}")
        if "STATE8_EARLY_LABEL_BINDING_WEIGHT" in env:
            if not str(env["REPAIR_BRANCH"]).startswith("early_label_"):
                raise ValueError(f"Early-label override on wrong branch: {name}")
            if float(env["STATE8_EARLY_LABEL_BINDING_WEIGHT"]) <= 0:
                raise ValueError(f"Early-label weight is not positive: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(args.config)
    print("valid=true")


if __name__ == "__main__":
    main()
