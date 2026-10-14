#!/usr/bin/env python3
"""Bind a failed-checkpoint repair plan to immutable Feeds jobs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_BRANCHES = {
    "balanced_continuation", "semantic_binding", "specificity_triplet",
    "identity_controls", "causal_teacher_low", "causal_teacher_high",
    "early_label_low", "early_label_high",
}
REQUIRED_OVERRIDES = {
    "STATE8_EEDI_CAUSAL_TWINS",
    "STATE8_CAUSAL_REPEAT_FRACTION",
    "STATE8_CAUSAL_WEIGHT",
    "STATE8_CAUSAL_MARGIN",
    "STATE8_EEDI_GEOMETRY_WEIGHT",
}
OPTIONAL_OVERRIDES = {
    "STATE8_CONTROL_RANDOM_SAMPLES",
    "STATE8_CONTROL_RANDOM_WEIGHT",
    "STATE8_CONTROL_SAME_RULE_WEIGHT",
    "STATE8_CAUSAL_TEACHER_TRAJECTORY_WEIGHT",
    "STATE8_CAUSAL_TEACHER_COSINE_WEIGHT",
    "STATE8_CAUSAL_TEACHER_RELATIVE_MSE_WEIGHT",
    "STATE8_EARLY_LABEL_BINDING_WEIGHT",
}
ALLOWED_OVERRIDES = REQUIRED_OVERRIDES | OPTIONAL_OVERRIDES

CURRICULUM_TAGS = {
    "generic": "state8_diagnosis_first_eedi_curriculum_20260812",
    "current_query": "state8_diagnosis_first_eedi_current_query_curriculum_20260812",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(
    family: str, arm: str, step: int, results_root: Path
) -> tuple[dict[str, Any], Path]:
    if family == "curriculum":
        if arm not in CURRICULUM_TAGS or step not in {2000, 3000, 4000, 5000}:
            raise ValueError(f"Invalid curriculum repair source: {arm}/{step}")
        path = (
            results_root
            / CURRICULUM_TAGS[arm]
            / "milestones"
            / f"checkpoint-{step}.completion.json"
        )
    elif family == "legacy":
        if arm not in {"updater", "full"} or step not in {20, 100}:
            raise ValueError(f"Invalid legacy repair source: {arm}/{step}")
        path = (
            results_root
            / f"state8_eedi_same_question_{arm}_checkpoint{step}_screen_20260811"
            / "checkpoint_completion.json"
        )
    else:
        raise ValueError(f"Unknown repair source family: {family}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or int(value.get("step", -1)) != step:
        raise ValueError(f"Incomplete checkpoint receipt: {path}")
    hashes = value.get("sha256") or {}
    if set(hashes) != {"adapter_model", "recurrent_state"}:
        raise ValueError(f"Checkpoint hash schema drift: {path}")
    for item in hashes.values():
        if len(item) != 64 or any(char not in "0123456789abcdef" for char in item):
            raise ValueError(f"Invalid checkpoint hash: {path}")
    return value, path


def materialize(template: Path, plan_path: Path, results_root: Path, output: Path) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("stage") not in {
        "state8_eedi_same_question_failed_checkpoint_repair_plan_v1",
        "state8_eedi_same_question_early_checkpoint_extension_plan_v1",
        "state8_curriculum_failed_checkpoint_repair_plan_v1",
    }:
        raise ValueError("Repair-plan stage drift")
    if plan.get("scientific_floors_unchanged") is not True:
        raise ValueError("Repair plan weakened scientific floors")
    if plan.get("raw_history_visible_to_decoder") is not False:
        raise ValueError("Repair plan opens decoder history")
    branches = plan.get("branches") or []
    if not 1 <= len(branches) <= 3:
        raise ValueError("Repair plan must contain one to three branches")
    seed_base = int(plan.get("seed_base", 4201))
    if not 4201 <= seed_base <= 9997 or seed_base + len(branches) - 1 > 9999:
        raise ValueError("Repair seed namespace is outside the audited range")

    config = yaml.safe_load(template.read_text(encoding="utf-8"))
    template_jobs = config.get("jobs") or []
    if len(template_jobs) != 1:
        raise ValueError("Template job matrix drift")
    base_job = template_jobs[0]
    wrapper = PROJECT_ROOT / "scripts/run_state8_eedi_same_question_repair_branch.sh"
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, branch in enumerate(branches):
        name = str(branch.get("name") or "")
        if name not in ALLOWED_BRANCHES or name in seen:
            raise ValueError(f"Invalid or duplicate repair branch: {name}")
        seen.add(name)
        source = branch.get("source") or {}
        family = str(source.get("family") or "legacy")
        arm, step = str(source.get("arm")), int(source.get("checkpoint_step", -1))
        training_arm = str(branch.get("training_arm") or "")
        if training_arm not in {"updater", "full"}:
            raise ValueError(f"Invalid training arm: {training_arm}")
        overrides = {str(k): str(v) for k, v in (branch.get("overrides") or {}).items()}
        if not REQUIRED_OVERRIDES <= set(overrides) <= ALLOWED_OVERRIDES:
            raise ValueError(f"Repair override schema drift: {sorted(overrides)}")
        receipt, receipt_path = _receipt(family, arm, step, results_root)
        hashes = receipt["sha256"]
        # Legacy plans retain their historical 3201 namespace.  Curriculum
        # plans default to 4201, while independently scheduled proactive plans
        # provide an explicit disjoint seed_base (currently 5201).
        seed = (seed_base if family == "curriculum" else 3201) + index
        job = copy.deepcopy(base_job)
        job["name"] = f"state8-eedi-repair-{name.replace('_', '-')}-seed{seed}"
        job["command"] = (
            "set -euo pipefail\n"
            "test \"$$(sha256sum scripts/run_state8_eedi_same_question_repair_branch.sh | cut -d' ' -f1)\" = "
            f"{digest(wrapper)}\n"
            "bash scripts/run_state8_eedi_same_question_repair_branch.sh\n"
        )
        env = job["submit_args"]["env"]
        env.clear()
        if family == "curriculum":
            source_arm = f"curriculum_{arm}"
            source_root = (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                f"{CURRICULUM_TAGS[arm]}/checkpoint-{step}"
            )
        else:
            source_arm = arm
            source_root = (
                "/mnt/amulet/studentmodeling/checkpoint_snapshots/"
                f"state8_eedi_same_question_20260811/{arm}/checkpoint-{step}"
            )
        env.update(
            _AZUREML_SINGULARITY_JOB_UAI="rankfun_aml",
            VENV_ROOT="/tmp/studentmodeling-state8-eedi-repair-venv",
            REPAIR_BRANCH=name,
            REPAIR_SEED=str(seed),
            SOURCE_CHECKPOINT_ARM=source_arm,
            SOURCE_CHECKPOINT_STEP=str(step),
            SOURCE_CHECKPOINT_ROOT=source_root,
            SOURCE_ANCHOR_MODE=("updater_mse1" if training_arm == "updater" else "full_mse1"),
            EXPECTED_STUDENT_ADAPTER_SHA256=hashes["adapter_model"],
            EXPECTED_STUDENT_STATE_SHA256=hashes["recurrent_state"],
            REPAIR_REMOTE_ROOT=(
                "/mnt/amulet/studentmodeling/diagnostics/"
                f"state8_stateonly_repair_seed{seed}_taskanchor_"
                f"source_anchor_{'updater' if training_arm == 'updater' else 'full'}_"
                f"mse1_eedi_same_question_v1_{name}_seed{seed}"
            ),
            **overrides,
        )
        jobs.append(job)
    config["description"] = (
        "Bound State8 same-question failed-checkpoint orthogonal repair branches"
    )
    config["code"]["local_dir"] = str(PROJECT_ROOT)
    config["environment"]["setup"] = [
        line.replace(
            "/tmp/studentmodeling-state8-eedi-same-question-repair-venv",
            "/tmp/studentmodeling-state8-eedi-repair-venv",
        )
        for line in config["environment"]["setup"]
    ]
    config["jobs"] = jobs
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.template, args.plan, args.results_root, args.output)
    print(f"materialized={args.output}")


if __name__ == "__main__":
    main()
