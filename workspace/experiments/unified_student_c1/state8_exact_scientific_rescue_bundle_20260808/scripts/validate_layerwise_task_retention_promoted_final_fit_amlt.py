#!/usr/bin/env python3
"""Fail-closed AMLT validator for the promoted task-retention exact fit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from amlt_validation_utils import command_text, target_matches, validate_shell_syntax


IMAGE = "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260615T232545258"
WORKSPACE = "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"
JOB = "layerwise-task-retention-promoted-finalfit-seed42"


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config: {path}")
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
    storage = (config.get("storage") or {}).get("student_store") or {}
    expected_storage = {
        "storage_account_name": "csnewsandfeeds4150361735",
        "container_name": "amulet",
        "mount_dir": "/mnt/amulet",
        "is_output": True,
    }
    if storage != expected_storage:
        raise ValueError("Storage drift")
    jobs = config.get("jobs") or []
    if len(jobs) != 1 or jobs[0].get("name") != JOB:
        raise ValueError("Job drift")
    job = jobs[0]
    if (
        job.get("sku") != "2x80G8-A100-IB-NvLink"
        or job.get("mpi") is not True
        or int(job.get("process_count_per_node", 0)) != -1
        or job.get("sla_tier") != "Standard"
    ):
        raise ValueError("GPU/MPI/SLA drift")
    submit = job.get("submit_args") or {}
    if (
        submit.get("max_attempts") != 2
        or submit.get("max_run_duration_seconds") != 345600
        or (submit.get("env") or {}).get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml"
        or (submit.get("env") or {}).get("VENV_ROOT")
        != "/tmp/studentmodeling-layerwise-promoted-final-venv"
    ):
        raise ValueError("Retry/UAI/venv drift")
    command = command_text(job.get("command"))
    validate_shell_syntax(command, JOB)
    if command.strip() != (
        "set -euo pipefail\n"
        "bash scripts/run_layerwise_task_retention_promoted_final_fit.sh"
    ):
        raise ValueError("Promoted final-fit launcher drift")

    script = (
        Path(__file__).resolve().parent
        / "run_layerwise_task_retention_promoted_final_fit.sh"
    ).read_text(encoding="utf-8")
    required = (
        'VENV_ROOT="${VENV_ROOT:-/tmp/studentmodeling-layerwise-promoted-final-venv}"',
        'python() { "$VENV_ROOT/bin/python" "$@"; }',
        "import accelerate,bitsandbytes,peft,safetensors,transformers",
        "ebf21ce4c0dbdf8b99c0a945d92aeec9d236a55d304b91912539034a8a6db81f",
        "58d974c91a04a9aee9271ef33ede187aafef84fd127ae08f15cc4dc64bb5d495",
        "c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708",
        "train_recurrent_student_m4_promoted.py",
        "20f85e37512965d560761de26d99d3329b15c16de679d34ad460a1407bad2f9a",
        "c80357282e600b31f68f55d8ab434ad349c0e9eeeeb222b52dbadae7caea11ca",
        "a995f82f364eb3247ff3e17873cc720bbce7a2a8f52813c9369421fa61d3c274",
        "task_retention_promotion/selected.json",
        "layerwise_task_retention_promoted_order_v1",
        "x['source_checkpoint_round'] == 750",
        "x['source_world_size'] == 16",
        "x['completed_prefix_rows'] == 12000",
        "x['epoch_rows'] == 259845",
        "expected_rows=(5000 if x['branch'] in {'eedi_full','eedi_full_open_t3'} else 2000 if x['branch']=='equal_t3_t6_open_t3_two_stage' else 1000)",
        "x['selected_loss_profile'] in {'standard','state_kd','state_path'}",
        "selected_learning_rate",
        "selected_causal_weight",
        "PROMOTION_LOSS_PROFILE",
        "PROMOTION_CANDIDATE",
        "train_recurrent_student_m4_promoted_open_t3.py",
        "train_recurrent_student_m4_promoted_open_t3_selective_kd.py",
        "train_recurrent_student_m4_promoted_open_t3_two_stage.py",
        "b52332455504ff5528989d8a71af2cd2edcd37a3425d8bedd03745d6248c3191",
        "3ee69fa74b056a6ca15774dacb52a1cfd402d4baf21b0d333a8e7d9f001c4650",
        "3fb17bffef5414655175165b26f67a7105d96243963bffd45f42102423b39ed5",
        "PROMOTION_BRANCH\" == equal_t3_t6_open_t3_two_stage",
        "standard) BEHAVIOR_KD_WEIGHT=0.5; DIRECT_REPLAY_WEIGHT=0.25",
        "state_kd) BEHAVIOR_KD_WEIGHT=0.5; DIRECT_REPLAY_WEIGHT=0.0",
        "state_path) BEHAVIOR_KD_WEIGHT=0.0; DIRECT_REPLAY_WEIGHT=0.0",
        "--checkpoint \"$CHECKPOINT/recurrent_state.pt\" --expected-round 750",
        "x['history_rows']==12000 and x['causal_exposures']==12992",
        "--state-architecture layerwise --layerwise-operation-checkpointing",
        "--layerwise-transition-rank 256",
        '--learning-rate "$ACTIVE_LEARNING_RATE" --lr-scheduler cosine --warmup-ratio 0.03',
        '--causal-weight "$PROMOTION_CAUSAL_WEIGHT"',
        "--causal-objective-mode specificity_triplet",
        "--causal-repeat-fraction 0.05",
        "--normalize-causal-exposure",
        "--promoted-real-order-manifest \"$PROMOTION\"",
        "--distributed-mpi --distributed-world-size 16",
        "--save-steps 250 --checkpoint-retention 2",
        "--recovery-checkpoint-first-offset 10 --recovery-checkpoint-interval 20",
        "select_latest_valid_distributed_checkpoint.py",
        "--expected-world-size 16 --max-round 16241 --save-steps 250",
        "--recovery-save-steps 10",
        "INITIAL_CHECKPOINT_SHA256=598901f8b8fc0901c6a1da6eaa32866dc1fd7dbfe762d34483b10dbaf23853fe",
        'local_resume_dir="/tmp/layerwise-promoted-resume/$resume_key"',
        "stage_distributed_resume_bundle.py",
        'stage_args=(--source-state "$resume_checkpoint"',
        'resume_args=(--resume-from-checkpoint "$local_resume")',
        "completion['micro_steps']==259845",
        "completion['real_examples_seen_once'] is True",
        "completion['exact_history_support'] is True",
        "completion['causal_exposure_count_matches_plan'] is True",
        "completion['all_causal_pairs_covered'] is True",
        'cp "$RECURRENT_SPLIT/audit.json" "$AMLT_OUTPUT_DIR/metrics/data_audit.json"',
        'cp "$RECURRENT_SPLIT/final_fit_manifest.json" "$AMLT_OUTPUT_DIR/metrics/final_fit_manifest.json"',
        "config['promoted_real_order_applied'] is True",
        "config['promoted_real_order_source_round']==750",
        "config['behavior_kd_weight'],config['direct_replay_weight']",
        "config['training_view']==expected_view",
        "raw_history_visible_to_decoder':False",
        "persistent_history_kv_cache':False",
        "/mnt/amulet/studentmodeling/models/Qwen3-32B",
        "/mnt/amulet/studentmodeling/checkpoints/layerwise_unified_world_model_seed42_taskretention_",
    )
    if any(fragment not in script for fragment in required):
        raise ValueError("Promoted final-fit command drift")
    trainer_path = Path(__file__).resolve().parent / "train_recurrent_student_m4_promoted.py"
    trainer = trainer_path.read_text(encoding="utf-8")
    import hashlib

    if hashlib.sha256(trainer_path.read_bytes()).hexdigest() != (
        "ebf21ce4c0dbdf8b99c0a945d92aeec9d236a55d304b91912539034a8a6db81f"
    ):
        raise ValueError("Promoted trainer SHA256 drift")
    cadence_path = Path(__file__).resolve().parent / "promoted_checkpoint_cadence.py"
    if hashlib.sha256(cadence_path.read_bytes()).hexdigest() != (
        "58d974c91a04a9aee9271ef33ede187aafef84fd127ae08f15cc4dc64bb5d495"
    ):
        raise ValueError("Promoted recovery checkpoint cadence SHA256 drift")
    stage_path = Path(__file__).resolve().parent / "stage_distributed_resume_bundle.py"
    if hashlib.sha256(stage_path.read_bytes()).hexdigest() != (
        "c89b26b4e5b69d60774e2fceed57e7c7a635d4b5d403e7e0f1e952127ada6708"
    ):
        raise ValueError("Distributed resume bundle stager SHA256 drift")
    trainer_required = (
        "def validate_promoted_resume_recipe(",
        "Promoted objective transition source weights drift",
        'if int(checkpoint_payload.get("round", -1)) != source_round:',
        'if source_weights != (0.5, 0.25):',
        'and selected_causal_weight == source_causal_weight',
        'normalized["behavior_kd_weight"]',
        'normalized["direct_replay_weight"]',
        'normalized["learning_rate"]',
        'normalized["causal_weight"]',
        "Promoted causal-weight transition target drift",
        "def apply_promoted_learning_rate_transition(",
        "def validate_two_stage_promotion(",
        "def apply_two_stage_consolidation_transition(",
        'TWO_STAGE_OPEN_T3_BRANCH = "equal_t3_t6_open_t3_two_stage"',
        "validate_distributed_resume_recipe(checkpoint_config, normalized)",
    )
    if any(fragment not in trainer for fragment in trainer_required):
        raise ValueError("Promoted objective-transition implementation drift")
    forbidden = (
        "password",
        "account_key",
        "access_token",
        "/home/v-yuejiali/studentmodeling/models",
        "--state-architecture scalar",
        "--diagnostic-warm-start-checkpoint",
        "--diagnostic-index-manifest",
        "--diagnostic-stop-after-round",
    )
    if any(fragment.lower() in script.lower() for fragment in forbidden):
        raise ValueError("Secret/local/scalar/diagnostic path leaked")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
