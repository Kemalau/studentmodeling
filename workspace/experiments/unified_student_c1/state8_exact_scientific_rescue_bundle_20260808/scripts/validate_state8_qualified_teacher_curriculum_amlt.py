#!/usr/bin/env python3
"""Fail-closed AMLT validator for gated Stage-B state curriculum."""

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
    "state8-qualified-teacher-curriculum-seed42": "t3_heavy",
    "state8-qualified-teacher-balanced-curriculum-seed42": "balanced",
    "state8-qualified-teacher-highkd-curriculum-seed42": "t3_heavy",
}
FILES = {
    "run_state8_qualified_teacher_curriculum.sh": "b545496b52c8aff82196c8ac917178b110e304b1ae4880ed24ca039cff626638",
    "train_recurrent_student_m4.py": "a67662448408a587f891d32d4c5170fdca0435adac47554268edf5a3972330b5",
    "train_recurrent_student_m4_state8_external_teacher_curriculum.py": "af1da212bc9c457a7b5a9297e895bb56fffcc7d98c8b2c9ff21ed0ae7f671539",
    "materialize_recurrent_index_manifest.py": "977ad3cf419f850b74979f430025eda3959ed3c7122aa5a2db5afff114f6b7df",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Invalid config")
    return value


def validate(config: dict) -> None:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if re.search(r"<placeholder>|<workspace-name>|<your-command>|\bREQUIRED\b", rendered):
        raise ValueError("Unresolved placeholder")
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
    jobs = config.get("jobs") or []
    job_names = {str(job.get("name")) for job in jobs}
    if not job_names or not job_names.issubset(JOBS) or len(job_names) != len(jobs):
        raise ValueError("Job drift")
    commands = []
    for job in jobs:
        name = str(job.get("name"))
        if (job.get("sku"), job.get("process_count_per_node"), job.get("mpi"), job.get("sla_tier")) != (
            "80G4-A100", -1, True, "Standard"
        ):
            raise ValueError("Topology drift")
        submit = job.get("submit_args") or {}
        env = submit.get("env") or {}
        if env.get("_AZUREML_SINGULARITY_JOB_UAI") != "rankfun_aml":
            raise ValueError("UAI drift")
        if env.get("STATE8_REPLAY_VARIANT") != JOBS[name]:
            raise ValueError("Replay variant drift")
        if str(env.get("STATE8_SAVE_STEPS")) != "125":
            raise ValueError("Checkpoint cadence drift")
        expected_kd_variant = (
            "highkd" if name == "state8-qualified-teacher-highkd-curriculum-seed42" else None
        )
        if env.get("STATE8_KD_VARIANT") != expected_kd_variant:
            raise ValueError("KD variant drift")
        command = command_text(job.get("command"))
        validate_shell_syntax(command, name)
        if "run_state8_qualified_teacher_curriculum.sh" not in command:
            raise ValueError("Entrypoint drift")
        commands.append(command)
    if len(set(commands)) != 1:
        raise ValueError("Curriculum arms changed more than replay variant")
    root = Path(__file__).resolve().parent
    for name, expected in FILES.items():
        observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"Source hash drift for {name}: {observed}")
    launcher = (root / "run_state8_qualified_teacher_curriculum.sh").read_text()
    for fragment in (
        "teacher_gate.json",
        "teacher_provenance_gate.json",
        "state8_clean_prompt_teacher_provenance_gate_v2",
        "legacy_answer_adjacent_contracts",
        "teacher_evaluation_input_sha256",
        "teacher_clean_prompt_sha256",
        "state8_natural_view_sha256",
        "tg['support']==1494 and tg['exact_match_is_authoritative'] is False",
        "pg['evaluation_shards']==16 and pg['evaluation_rows']==1494",
        "--state-token-count 8",
        'STATE8_STUDENT_INIT_CAPACITY="${STATE8_STUDENT_INIT_CAPACITY:-8}"',
        'STATE8_STUDENT_STATE_PATH:-$STUDENT_ROOT/model/recurrent_state.pt',
        "int(config.get('state_token_count',1))==expected_capacity",
        '--init-recurrent-state "$STUDENT_STATE_PATH"',
        '--init-adapter "$STUDENT_MODEL_DIR"',
        '--learning-rate "$STATE8_LEARNING_RATE"',
        "--history-dropout-curriculum --dual-path-curriculum",
        "--direct-offload-min-tokens 1",
        '--online-kd-weight "$STATE8_ONLINE_KD_WEIGHT"',
        "--consistency-weight 0.0 --direct-replay-weight 0.0",
        "--causal-objective-mode specificity_triplet",
        '--save-steps "$STATE8_SAVE_STEPS" --recovery-save-steps 25',
        '--save-steps "$STATE8_SAVE_STEPS" --checkpoint-retention 2',
        "--twins 125 --seed 42",
        "x['pairs']==250 and x['twins']==125",
        "STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_v3",
        "STATE8_STAGE_B_RECIPE_ID=exact5000_rounds1250_kd1_v2",
        "STATE8_ONLINE_KD_WEIGHT=0.5",
        "STATE8_ONLINE_KD_WEIGHT=1.0",
        "state8_qualified_teacher_curriculum_seed42_${STATE8_STAGE_B_RECIPE_ID}_20260805",
        "STATE8_REPLAY_VARIANT",
        "state8_qualified_balanced_replay_seed42_v1.json",
        "baa0c9f7c062822cfffe11a5a4ec1e416bf726dfaf66fd8251f7dd6bb03bbadd",
        "338ca9e62da39e03e8e2a25dec56c57179a4206ebf1dfef9729042b1ea681deb",
        "state8_qualified_teacher_balanced_curriculum_seed42_${STATE8_STAGE_B_RECIPE_ID}_20260805",
        "state8_qualified_teacher_balanced_curriculum_v1_immutable",
        'cp "$STABLE_ROOT/metrics/immutable_receipt.json"',
        "counts=={0.0:1000,0.25:1000,0.5:1000,0.75:1000,1.0:1000}",
        "raw_history_visible_to_decoder':False",
        "direct_offload_min_tokens",
        "023e692afe932a6bbbd70547b698007c89b96ae7f7d79adae4622fdf34900540",
        "state8_qualified_teacher_curriculum_v1_immutable",
        "evaluate_recurrent_formal_protocols.py",
        "formal_screen_rows':487",
    ):
        if fragment not in launcher:
            raise ValueError(f"Stage-B launcher drift: {fragment}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    validate(load(args.config))
    print(f"validated={args.config}")


if __name__ == "__main__":
    main()
