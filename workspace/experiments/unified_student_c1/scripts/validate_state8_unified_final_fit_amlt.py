#!/usr/bin/env python3
"""Fail-closed preflight for the exact state8 unified final fit."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOBS = {
    "state8-qualified-unified-finalfit-seed42": "t3_heavy",
    "state8-qualified-balanced-unified-finalfit-seed42": "balanced",
    "state8-qualified-highkd-unified-finalfit-seed42": "highkd",
    "state8-repair-logits-unified-finalfit-seed42": "repair_logits",
    "state8-repair-trajectory01-unified-finalfit-seed42": "repair_trajectory01",
    "state8-semantic-c4-g005-unified-finalfit-seed42": "semantic_c4_g005",
    "state8-semantic-c4-g010-unified-finalfit-seed42": "semantic_c4_g010",
    "state8-semantic-c4-g025-unified-finalfit-seed42": "semantic_c4_g025",
    "state8-semantic-eedi-g005-unified-finalfit-seed42": "semantic_eedi_g005",
    "state8-semantic-eedi-g010-unified-finalfit-seed42": "semantic_eedi_g010",
    "state8-semantic-eedi-g025-unified-finalfit-seed42": "semantic_eedi_g025",
    "state8-semantic-eedi-g050-unified-finalfit-seed42": "semantic_eedi_g050",
    "state8-semantic-eedi-g025-single-seed46-unified-finalfit-seed42": "semantic_eedi_g025_single_seed46",
    "state8-highcausal-joint-unified-finalfit-seed42": "highcausal_joint",
    "state8-highcausal-updater-unified-finalfit-seed42": "highcausal_updater",
    "state8-highcausal-mid-unified-finalfit-seed42": "highcausal_mid",
    "state8-highcausal-eedi-unified-finalfit-seed42": "highcausal_eedi",
    "state8-highcausal-taxonomy-unified-finalfit-seed42": "highcausal_taxonomy",
    "state8-taxonomy-curriculum-t3heavy-unified-finalfit-seed42": "taxonomy_curriculum_t3heavy",
    "state8-taxonomy-curriculum-balanced-unified-finalfit-seed42": "taxonomy_curriculum_balanced",
    "state8-taxonomy-curriculum-t3heavy-diag-unified-finalfit-seed42": "taxonomy_curriculum_t3heavy_diag",
    "state8-taxonomy-curriculum-balanced-diag-unified-finalfit-seed42": "taxonomy_curriculum_balanced_diag",
    "state8-task-anchor-base-unified-finalfit-seed42": "task_anchor_base",
    "state8-task-anchor-c4strong-unified-finalfit-seed42": "task_anchor_c4strong",
    "state8-strong1-widen8-lr5e7-seed58-unified-finalfit-seed42": "strong1_widen8_lr5e7_seed58",
    "state8-strong1-widen8-lr1e6-seed59-unified-finalfit-seed42": "strong1_widen8_lr1e6_seed59",
    "state8-strong1-widen8-single-lr5e7-seed60-unified-finalfit-seed42": "strong1_widen8_single_lr5e7_seed60",
    "state8-strong1-rescue-taxonomy-joint-seed61-unified-finalfit-seed42": "strong1_rescue_taxonomy_joint_seed61",
    "state8-strong1-rescue-taskanchor-c4strong-seed62-unified-finalfit-seed42": "strong1_rescue_taskanchor_c4strong_seed62",
    "state8-strong1-midpoint-diagnosis-first-seed63-unified-finalfit-seed42": "strong1_midpoint_diagnosis_first_seed63",
    "state8-strong1-exact-diagnosis-curriculum-seed65-unified-finalfit-seed42": "strong1_exact_diagnosis_curriculum_seed65",
    "state8-strong1-exact-rescue-semantic-preserve-seed66-unified-finalfit-seed42": "strong1_exact_rescue_semantic_preserve_seed66",
    "state8-strong1-exact-rescue-causal-taskanchor-seed67-unified-finalfit-seed42": "strong1_exact_rescue_causal_taskanchor_seed67",
    "state8-strong1-exact-rescue-retention-preserve-seed68-unified-finalfit-seed42": "strong1_exact_rescue_retention_preserve_seed68",
    "state8-diagnosis-first-matched-seed67-unified-finalfit-seed42": "diagnosis_first_matched_seed67",
}
FILES = {
    "train_recurrent_student_m4.py": "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
    "train_recurrent_student_m4_state8_external_teacher_curriculum.py": "af1da212bc9c457a7b5a9297e895bb56fffcc7d98c8b2c9ff21ed0ae7f671539",
    "train_state8_stateonly_teacher_repair.py": "3f8e05a6b35515c52b53482f894afbaedca6de4ce163a53b0737aa8971637452",
    "train_state8_semantic_causal_repair.py": "14db1f4c430c61537a33c0b4617f1734a0bd51501249dfa8f61bc8c0ec01759a",
    "train_state8_task_anchored_causal_repair.py": "b622a62fffef45a19b3b02b771cee4412d5f9d17f1a3fe99c6ba046e6111e66b",
    "train_state8_diagnosis_first_causal_repair.py": "7d638013f932f29f52cffeea709b96872df8ba0960b9bb4e56ed23cc3475b216",
    "train_state8_diagnosis_first_task_anchored_causal_repair.py": "eb9837127af9fe3761f6fa57b78a3e9ba3373a223706f0a4921749eab61e35a3",
    "train_state8_diagnosis_first_retention_anchored_causal_repair.py": "c8e4e2bebf7ebf01b15685b74a5bdcbaa0afcb5a16c774b40ba0d7070819095a",
    "filter_state8_causal_diagnosis.py": "ba04c7d3cd1e74843678d2b3e66a0dea7c0aac647e7b149f18fc1f7b3bebbc58",
    "augment_state8_eedi_semantic_pairs.py": "c9006ac322d075ea1fa35ee9ace3a6561f799bdfea767786a6e459825b0ad46c",
    "state8_natural_curriculum_view.py": "98c961180d2428b2c68ef9399996ed991d5bd34120a3046b4f56fb9ba4e09682",
    "augment_recurrent_causal_diagnosis.py": "b74684001e54bf0c9b023540c76a269708f1e8fa529cfec703a4439288590926",
    "publish_state8_final_checkpoint.py": None,
    "run_state8_unified_final_fit.sh": None,
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid AMLT config")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
    if "msrcommonacr.azurecr.io/ml-common-py3.12:latest" in rendered:
        raise ValueError("Forbidden image")
    if not target_matches(config.get("target"), WORKSPACE):
        raise ValueError("Target drift")
    if (config.get("environment") or {}).get("image") != IMAGE:
        raise ValueError("Image drift")
    if (config.get("storage") or {}).get("student_store") != {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }:
        raise ValueError("Storage drift")
    code = config.get("code") or {}
    if not (
        str(code.get("local_dir")) == "$CONFIG_DIR/.."
        or (code.get("local_dir") is None and code.get("storage_id") == "default")
    ):
        raise ValueError("Code root drift")
    if not {".git/", "data/", "results/"}.issubset(set(code.get("ignore") or [])):
        raise ValueError("Code exclusions drift")
    jobs = config.get("jobs") or []
    job_names = {str(job.get("name")) for job in jobs}
    if not job_names or not job_names.issubset(JOBS) or len(job_names) != len(jobs):
        raise ValueError("Job set drift")
    commands = []
    for job in jobs:
        name = str(job.get("name"))
        expected_world4 = JOBS[name].startswith(("strong1_widen8_", "strong1_rescue_", "strong1_midpoint_", "strong1_exact_"))
        expected_sku = "80G4-A100" if expected_world4 else "2x80G8-A100-IB-NvLink"
        expected_launcher = (
            "bash scripts/run_state8_unified_final_fit_world4.sh"
            if expected_world4
            else "bash scripts/run_state8_unified_final_fit.sh"
        )
        if (
            job.get("sku") != expected_sku
            or int(job.get("process_count_per_node", 0)) != -1
            or job.get("mpi") is not True
            or job.get("sla_tier") != "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}; env = submit.get("env") or {}
        expected_env = {
            "_AZUREML_SINGULARITY_JOB_UAI": "rankfun_aml",
            "VENV_ROOT": "/tmp/studentmodeling-state8-finalfit-venv",
            "FINALFIT_VARIANT": JOBS[name],
        }
        if JOBS[name].startswith(("strong1_widen8_", "strong1_rescue_", "strong1_midpoint_", "strong1_exact_")):
            expected_env["STATE8_CANONICAL_RUN_DIR"] = (
                "/mnt/amulet/studentmodeling/checkpoints/"
                "state8_strong1_widen8_unified_full259845_seed42_v1_20260808"
            )
        if env != expected_env:
            raise ValueError("Environment/UAI/variant drift")
        expected_duration = (
            43200
            if JOBS[name].startswith(("strong1_widen8_", "strong1_rescue_", "strong1_midpoint_", "strong1_exact_"))
            else 345600
        )
        if submit.get("max_attempts") != 2 or submit.get("max_run_duration_seconds") != expected_duration:
            raise ValueError("Retry/duration drift")
        command = command_text(job.get("command")); validate_shell_syntax(command, name)
        for fragment in ("set -euo pipefail", "FAILED line=%s rc=%s command=%s", expected_launcher):
            if fragment not in command: raise ValueError(f"Launcher drift: {fragment}")
        commands.append(command)
    expected_command_count = 1 if all(
        JOBS[str(job.get("name"))].startswith(("strong1_widen8_", "strong1_rescue_", "strong1_midpoint_", "strong1_exact_"))
        for job in jobs
    ) or all(
        not JOBS[str(job.get("name"))].startswith(("strong1_widen8_", "strong1_rescue_", "strong1_midpoint_", "strong1_exact_"))
        for job in jobs
    ) else 2
    if len(set(commands)) != expected_command_count:
        raise ValueError("Final-fit arms changed beyond the approved topology families")
    root = Path(__file__).resolve().parent
    launcher = (root / "run_state8_unified_final_fit.sh").read_text(encoding="utf-8")
    for fragment in (
        "state8_qualified_teacher_curriculum_v1_immutable",
        "state8_qualified_teacher_balanced_curriculum_v1_immutable",
        "state8_qualified_teacher_curriculum_highkd_v1_immutable",
        "state8_stateonly_repair_seed44_logits_v1",
        "state8_stateonly_repair_seed44_trajectory01_v1",
        "semantic_c4_g005|semantic_c4_g010|semantic_c4_g025|semantic_eedi_g005|semantic_eedi_g010|semantic_eedi_g025|semantic_eedi_g050|semantic_eedi_g025_single_seed46",
        "train_state8_semantic_causal_repair.py",
        "augment_state8_eedi_semantic_pairs.py",
        "STATE8_RANDOM_NEUTRAL=1",
        "STATE8_EEDI_GEOMETRY_WEIGHT",
        "canonical_labels_visible_to_decoder",
        "semantic_random_target",
        "FINALFIT_VARIANT",
        "state8_full_history_natural_teacher_v1",
        "state8_qualified_unified_full259845_seed42_${FINALFIT_VARIANT}_v1_20260805",
        "state8_qualified_unified_full259845_seed42_v1_20260805",
        "publish_state8_final_checkpoint.py",
        'test -f "$CANONICAL_RUN_DIR/canonical_publish_receipt.json"',
        "state8_independent_full_history_teacher_gate_v1",
        "exact_match_is_authoritative",
        "STUDENTMODELING_EXTERNAL_STATE_TEACHER_SHA256",
        "train_recurrent_student_m4_state8_external_teacher_curriculum.py",
        "train_state8_stateonly_teacher_repair.py",
        "EXPECTED_HISTORY_MODE=strict_state_only",
        "STATE8_REPAIR_LABEL_WEIGHT=4.0",
        "STATE8_REPAIR_TRAJECTORY_WEIGHT=0.1",
        "--state-token-count 8",
        "--history-dropout-curriculum --dual-path-curriculum",
        "--direct-offload-min-tokens 1",
        '--online-kd-weight "$STATE8_FINALFIT_ONLINE_KD_WEIGHT"',
        '--learning-rate "$STATE8_FINALFIT_LEARNING_RATE"',
        '--consistency-weight "$STATE8_REPAIR_TRAJECTORY_WEIGHT"',
        "--causal-objective-mode specificity_triplet",
        "augment_recurrent_causal_diagnosis.py",
        "causal_answer_diagnosis_augmentation.json",
        "diagnosis_natural_schema",
        "canonical_label_auxiliary_supervision",
        '--causal-repeat-fraction "$STATE8_CAUSAL_REPEAT_FRACTION"',
        "--checkpoint-milestone-rounds $MILESTONE_ROUNDS",
        "--expected-world-size 16",
        '--expected-planned-optimizer-steps "$TOTAL_ROUNDS"',
        "--distributed-world-size 16",
        "--save-steps 50 --recovery-save-steps 25",
        "--save-steps 50 --checkpoint-retention 2",
        "completion['micro_steps']==259845",
        "history_dropout_counts",
        "direct_offload_min_tokens",
        "state8_qualified_unified_final_fit_v1",
        "teacher_adapter_sha256",
        "teacher_gate_required",
        "state8_clean_prompt_teacher_provenance_gate_v2",
        "teacher_evaluation_input_sha256",
        "teacher_clean_prompt_sha256",
        "legacy_answer_adjacent_contracts",
        "selected_replay_variant",
        "taxonomy_curriculum_t3heavy|taxonomy_curriculum_balanced|taxonomy_curriculum_t3heavy_diag|taxonomy_curriculum_balanced_diag",
        "state8_unified_candidate_selection_v1",
        "task_anchor_base|task_anchor_c4strong",
        "strong1_widen8_lr5e7_seed58|strong1_widen8_lr1e6_seed59|strong1_widen8_single_lr5e7_seed60",
        "state8_strong1_widen8_joint_promotion_v1",
        "strong1_rescue_taxonomy_joint_seed61|strong1_rescue_taskanchor_c4strong_seed62",
        "state8_strong1_rescue_joint_promotion_v1",
        "strong1_midpoint_diagnosis_first_seed63",
        "state8_midpoint_diagnosis_first_joint_promotion_v1",
        "strong1_exact_diagnosis_curriculum_seed65",
        "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "strong1_exact_rescue_semantic_preserve_seed66|strong1_exact_rescue_causal_taskanchor_seed67|strong1_exact_rescue_retention_preserve_seed68",
        "state8_exact_scientific_rescue_joint_promotion_v1",
        "train_state8_diagnosis_first_task_anchored_causal_repair.py",
        "train_state8_diagnosis_first_retention_anchored_causal_repair.py",
        "diagnosis_first_task_anchored_causal_repair.json",
        "train_state8_task_anchored_causal_repair.py",
        "task_anchored_causal_repair.json",
        "x['selected']['artifact_sha256']==observed",
        "INIT_HISTORY_MODE",
        "1c8916da482c316dee92982aa3afaa0d2a123f1574140789b14ba5e1e92aee95",
        "[64,8,5120]",
    ):
        if fragment not in launcher:
            raise ValueError(f"Final-fit invariant drift: {fragment}")
    for name, expected in FILES.items():
        if expected is None:
            continue
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    if any(JOBS[str(job.get("name"))].startswith(("strong1_widen8_", "strong1_rescue_", "strong1_midpoint_", "strong1_exact_")) for job in jobs):
        world4_wrapper = (root / "run_state8_unified_final_fit_world4.sh").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "render_state8_finalfit_world4_launcher.py",
            "c8389d874c51af0c9a809607d6b795fd22b7c125e51f58d245771e97aa73f6f0",
            "18b00cd75ea9f50a61806c94815e5b103d6fc81b23a5d79988894d1985bfcd75",
        ):
            if fragment not in world4_wrapper:
                raise ValueError(f"World4 wrapper drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
