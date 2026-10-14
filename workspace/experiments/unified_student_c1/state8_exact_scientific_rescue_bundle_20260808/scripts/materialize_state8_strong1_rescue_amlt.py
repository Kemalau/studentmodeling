#!/usr/bin/env python3
"""Materialize a hash-bound two-arm single-A100 rescue config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
ARMS: dict[str, dict[str, str]] = {
    "taxonomy_joint": {
        "seed": "61",
        "entrypoint": "scripts/train_state8_diagnosis_first_causal_repair.py",
        "repair_arm": "trajectory01",
        "tag": "strong1_rescue_taxonomy_joint_v1",
        "repeat": "0.50",
        "causal_weight": "2.0",
        "trajectory": "0.1",
        "label_weight": "1.0",
    },
    "taskanchor_c4strong": {
        "seed": "62",
        "entrypoint": "scripts/train_state8_task_anchored_causal_repair.py",
        "repair_arm": "taskanchor",
        "tag": "strong1_rescue_taskanchor_c4strong_v1",
        "repeat": "1.0",
        "causal_weight": "4.0",
        "trajectory": "1.0",
        "label_weight": "4.0",
    },
}


def load_selection(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("stage") != "state8_strong1_rescue_source_selection_v1":
        raise ValueError("Rescue source selection stage drift")
    selected = value.get("selected") or {}
    slug = str(selected.get("slug") or "")
    root = str(selected.get("remote_root") or "")
    artifacts = selected.get("artifact_sha256") or {}
    if slug not in {"lr5e7_seed58", "lr1e6_seed59", "single_lr5e7_seed60"}:
        raise ValueError("Rescue source slug drift")
    suffix = slug + ("_single_ga4" if slug == "single_lr5e7_seed60" else "")
    expected_root = (
        "/mnt/amulet/studentmodeling/diagnostics/"
        "state8_qualified_teacher_curriculum_highkd_v1_immutable_"
        "taxonomy_semantic_v1_strong1_widen8_" + suffix
    )
    if root != expected_root:
        raise ValueError(f"Rescue source root drift: {root}")
    if set(artifacts) != {"adapter_model", "recurrent_state"} or any(
        not isinstance(item, str) or len(item) != 64 for item in artifacts.values()
    ):
        raise ValueError("Rescue source artifact drift")
    return value


def materialize(selection_path: Path, output: Path) -> None:
    value = load_selection(selection_path)
    selected = value["selected"]
    slug = str(selected["slug"])
    root = str(selected["remote_root"])
    artifacts = selected["artifact_sha256"]
    common = {
        "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
        "VENV_ROOT": "/tmp/studentmodeling-state8-strong1-rescue-venv",
        "REQUESTED_WORLD_SIZE": "1",
        "GRADIENT_ACCUMULATION_STEPS": "4",
        "STATE8_REPAIR_SAVE_STEPS": "500",
        "STATE8_REPAIR_RECOVERY_SAVE_STEPS": "500",
        "STUDENT_ROOT": root,
        "EXPECTED_STUDENT_ADAPTER_SHA256": artifacts["adapter_model"],
        "EXPECTED_STUDENT_STATE_SHA256": artifacts["recurrent_state"],
        "STRONG1_SOURCE_SLUG": slug,
        "STATE8_T3_ANCHOR_SCHEDULE": "1.0,0.5,0.25",
        "STATE8_OTHER_ANCHOR_WEIGHT": "0.1",
        "STATE8_RANDOM_NEUTRAL": "1",
        "STATE8_GEOMETRY_WEIGHT": "0.10",
        "STATE8_EEDI_GEOMETRY_WEIGHT": "0.05",
        "STATE8_DIFFERENT_MAX_COSINE": "0.90",
        "STATE8_CAUSAL_DIAGNOSIS_ONLY": "0",
        "STATE8_CAUSAL_MARGIN": "0.5",
        "STATE8_LAYERWISE_TRANSITION_ONLY": "0",
        "STATE8_UPDATER_ONLY": "0",
        "STATE8_FORMAL_SCOPE": "unified",
        "STATE8_REPAIR_MANIFEST": "manifests/layerwise_task_retention_staged_t3_t6_seed42_v1.json",
        "STATE8_REPAIR_MANIFEST_SHA256": "a4eb41b2130f7e0810eeff36b5f3d16b8dd2c8631c50dbd25216f0793af23557",
    }
    jobs = []
    for arm, settings in ARMS.items():
        jobs.append(
            {
                "name": f"state8-strong1-rescue-{arm}-seed{settings['seed']}",
                "sku": "80G1-A100",
                "process_count_per_node": 1,
                "mpi": False,
                "sla_tier": "Standard",
                "command": (
                    "set -euo pipefail\n"
                    "trap 'rc=$$?; printf \"FAILED line=%s rc=%s command=%s\\n\" "
                    '"$${LINENO}" "$$rc" "$${BASH_COMMAND}" >&2; exit "$$rc"\' ERR\n'
                    "bash scripts/run_state8_strong1_rescue.sh\n"
                ),
                "submit_args": {
                    "max_attempts": 2,
                    "max_run_duration_seconds": 345600,
                    "env": {
                        **common,
                        "REPAIR_ARM": settings["repair_arm"],
                        "TRAIN_ENTRYPOINT": settings["entrypoint"],
                        "RUN_TAG": settings["tag"],
                        "TRAIN_SEED": settings["seed"],
                        "STATE8_CAUSAL_REPEAT_FRACTION": settings["repeat"],
                        "STATE8_CAUSAL_WEIGHT": settings["causal_weight"],
                        "STATE8_CAUSAL_LABEL_WEIGHT": settings["label_weight"],
                        "STATE8_REPAIR_LABEL_WEIGHT": settings["label_weight"],
                        "STATE8_DIAGNOSIS_FIRST_ENTRYPOINT_SHA256": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
                        "STATE8_DIAGNOSIS_FIRST_FORMAL_SHA256": "a8bc07a8fe6d2c9bfa9af6d976b53b84321cb4baa9c9fcace3188e370f7732c9",
                        "STATE8_DIAGNOSIS_FIRST_C4_SHA256": "7eef90e079d698c94f1f8655a1df3fe6bd93cedbb175ba4e7082a49bb8069c8b",
                    },
                },
            }
        )
    config = {
        "description": (
            f"Hash-bound strict-state-only strong1 rescue from {slug}: "
            "diagnosis-first semantic rescue versus task-anchored C4-strong"
        ),
        "target": {"service": "manifold", "name": "Feeds", "workspace_name": WORKSPACE},
        "environment": {
            "image": IMAGE,
            "setup": [
                "python -m venv --system-site-packages /tmp/studentmodeling-state8-strong1-rescue-venv",
                "/tmp/studentmodeling-state8-strong1-rescue-venv/bin/python -m pip install --upgrade pip",
                "/tmp/studentmodeling-state8-strong1-rescue-venv/bin/python -m pip install -r requirements-feeds.txt",
            ],
        },
        "code": {
            "local_dir": "$CONFIG_DIR/../strong1_rescue_bundle_20260808",
            "ignore": [".git/", ".vscode/", "amlt/", "data/", "results/", "__pycache__/", "*.pyc"],
        },
        "storage": {
            "student_store": {
                "storage_account_name": "csnewsandfeeds4150361735",
                "container_name": "amulet",
                "mount_dir": "/mnt/amulet",
                "is_output": True,
            }
        },
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.selection, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
