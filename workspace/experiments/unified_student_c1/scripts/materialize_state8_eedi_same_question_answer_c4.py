#!/usr/bin/env python3
"""Bind the approved answer-C4 template to a same-question selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def materialize(template: Path, selection_path: Path, output: Path) -> None:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    stage = selection.get("stage")
    if stage not in {
        "state8_eedi_same_question_joint_diagnostic_selection_v1",
        "state8_curriculum_checkpoint_joint_selection_v1",
        "state8_global_joint_diagnostic_selection_v1",
    }:
        raise ValueError("Selection stage drift")
    selected = selection.get("selected") or {}
    arm = selection.get("selected_arm")
    old_arm = isinstance(arm, str) and (
        arm in {"updater", "full"} or arm.startswith("repair_")
    )
    curriculum_arm = isinstance(arm, str) and arm.startswith("curriculum_")
    repair_checkpoint = selected.get("artifact_layout") == "repair_checkpoint_snapshot"
    if selection.get("pass") is not True or not (old_arm or curriculum_arm):
        raise ValueError("No qualified same-question arm")
    artifacts = selected.get("artifact_sha256") or {}
    if set(artifacts) != {"adapter_model", "recurrent_state"}:
        raise ValueError("Artifact hash drift")
    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    jobs = config.get("jobs") or []
    if len(jobs) != 1:
        raise ValueError("Template job matrix drift")
    job = jobs[0]
    job["name"] = "state8-eedi-same-question-answer-c4-selected"
    runner = Path(__file__).with_name("run_state8_eedi_same_question_answer_c4.sh")
    job["command"] = (
        "set -euo pipefail\n"
        f"test \"$$(sha256sum scripts/{runner.name} | cut -d' ' -f1)\" = "
        f"{hashlib.sha256(runner.read_bytes()).hexdigest()}\n"
        f"bash scripts/{runner.name}\n"
    )
    env = (job.setdefault("submit_args", {})).setdefault("env", {})
    env.clear()
    env.update(
        _AZUREML_SINGULARITY_JOB_UAI="rankfun_aml",
        VENV_ROOT="/tmp/studentmodeling-state8-eedi-same-question-answer-c4-venv",
        SAME_QUESTION_ARM=arm,
        SAME_QUESTION_REMOTE_ROOT=str(selected.get("remote_root") or ""),
        SAME_QUESTION_ARTIFACT_LAYOUT=(
            "repair_checkpoint_snapshot"
            if repair_checkpoint
            else ("curriculum_snapshot" if curriculum_arm else "repair_publish")
        ),
        SAME_QUESTION_ADAPTER_SHA256=artifacts["adapter_model"],
        SAME_QUESTION_STATE_SHA256=artifacts["recurrent_state"],
    )
    valid_root = (
        env["SAME_QUESTION_REMOTE_ROOT"].startswith(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_curriculum_repairs_20260812/"
        )
        if repair_checkpoint
        else (
        env["SAME_QUESTION_REMOTE_ROOT"].startswith(
            "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
            "state8_diagnosis_first_eedi_"
        )
        if curriculum_arm
        else env["SAME_QUESTION_REMOTE_ROOT"].startswith(
            "/mnt/amulet/studentmodeling/diagnostics/state8_stateonly_repair_seed"
        )
        )
    )
    if not valid_root:
        raise ValueError("Selected candidate remote root drift")
    config["description"] = "Frozen 150-pair answer-C4 for selected identical-query Eedi State8 candidate"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.selection, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
