#!/usr/bin/env python3
"""Build a local, provenance-aware live status without inventing remote state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitor_layerwise_unified_training_progress import health_summary


CONTROLLERS = (
    "final_fit",
    "progress",
    "promoted_progress",
    "checkpoint500_c4",
    "checkpoint500_causal_rescue_recovery",
    "known_oom_row_alignedmask",
    "task_retention_rehearsal",
    "eedi_state_path_ab",
    "state_kd",
    "equal_state_path",
    "equal_state_kd",
    "equal_state_kd_open_t3",
    "equal_state_path_open_t3",
    "equal_state_kd_open_t3_lr_scan",
    "equal_state_kd_open_t3_lr1e6",
    "open_t3_consolidation",
    "staged_task_retention",
    "staged_state_path",
    "eedi_full_lr_scan",
    "eedi_full_state_kd_lr_scan",
    "eedi_full_open_t3_lr_scan",
    "eedi_full_open_t3_selective_kd_lr_scan",
    "eedi_full_open_t3_selective_kd_lr_scan/local_lr2e6_round3800_preflight",
    "eedi_full_open_t3_selective_kd_lr_scan/local_lr2e6_round3950_c4",
    "eedi_open_t3_t6_repair",
    "task_retention_promoted_final_fit",
    "task_retention_promoted_final_fit/local_round1000_c4",
    "task_retention_promoted_final_fit/round1000_c4_n150_multiseed",
    "task_retention_promoted_final_fit/local_round1000_t3_semantic",
    "t3_full_candidate_screen_20260804/lr5e6_statepath",
    "cf1_fulltask_crossadapter_full_t3_20260804",
    "selective_kd_lr2e6_full_t3_20260804",
    "current_lr2e6_fulltask_crossadapter_full_t3_20260804",
    "current_lr2e6_direct_teacher_t3_20260804",
    "t3_direct_teacher_crossmatrix_eedi75_20260804",
    "external_state_teacher_eedi_full_20260804",
    "gold_semantic_teacher_eedi_full_20260804",
    "task_retention_promoted_final_eval",
    "task_retention_promoted_t3_semantic_judge",
    "checkpoint750_local_audit/formal_full",
    "checkpoint1000_c4",
    "checkpoint1000_formal_screen",
    "checkpoint1000_t3_semantic_judge",
    "checkpoint1000_decision",
    "final_eval",
    "t3_semantic_judge",
)


def timestamp(path: Path) -> datetime | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def text(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def summarize(root: Path, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    supervisor = root.parent / "recurrent_unified_supervisor_20260731"
    auth_required = (supervisor / "auth_required").is_file()
    legacy_experiment = text(root / "final_fit/active_experiment.txt")
    legacy_superseded = (root / "final_fit/superseded").is_file()
    promoted_experiment = text(
        root / "task_retention_promoted_final_fit/active_experiment.txt"
    )
    active_experiment = promoted_experiment or (
        legacy_experiment if not legacy_superseded else None
    )
    progress_root = root / (
        "promoted_progress" if promoted_experiment else "progress"
    )
    progress = load(progress_root / "progress.json")
    progress_experiment = progress.get("experiment") if progress else None
    progress_matches_active = bool(
        progress and active_experiment and progress_experiment == active_experiment
    )
    current_progress = progress if progress_matches_active else None
    awaiting_first_record = bool(
        current_progress
        and current_progress.get("phase") == "awaiting_first_training_record"
    )
    health = (current_progress.get("health") or {}) if current_progress else {}
    health_source = "progress_receipt" if health else None
    if not health and progress_matches_active and not awaiting_first_record:
        last_log = progress_root / "latest_rank0.log"
        if last_log.is_file():
            try:
                health = health_summary(
                    last_log.read_text(encoding="utf-8", errors="replace")
                )
                health_source = "last_downloaded_rank0_log"
            except ValueError:
                health = {}
    observed = None
    if current_progress and current_progress.get("observed_at"):
        observed = datetime.fromisoformat(str(current_progress["observed_at"]))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
    progress_age = (now - observed).total_seconds() if observed else None
    progress_fresh = progress_age is not None and 0 <= progress_age <= 900
    controllers: dict[str, Any] = {}
    for name in CONTROLLERS:
        directory = root / name
        heartbeat = timestamp(directory / "controller.heartbeat")
        age = (now - heartbeat).total_seconds() if heartbeat else None
        controllers[name] = {
            "heartbeat": heartbeat.isoformat() if heartbeat else None,
            "heartbeat_age_seconds": age,
            "alive_locally": age is not None and 0 <= age <= 900,
            "needs_attention": (directory / "needs_attention").is_file(),
            "complete": (directory / "complete").is_file(),
            "superseded": (directory / "superseded").is_file(),
            "deferred": (directory / "deferred").is_file(),
        }
        controllers[name]["required_now"] = not any(
            controllers[name][marker]
            for marker in ("complete", "superseded", "deferred")
        )
    attention = sorted(
        name
        for name, row in controllers.items()
        if row["needs_attention"] and not row["superseded"]
    )
    completion = load(root / "overall_completion.json")
    rehearsal_complete = (root / "task_retention_rehearsal/complete").is_file()
    promoted_fit_complete = (
        root / "task_retention_promoted_final_fit/complete"
    ).is_file()
    promoted_fit_deferred = (
        root / "task_retention_promoted_final_fit/deferred"
    ).is_file()
    round1000_multiseed = load(
        root
        / "task_retention_promoted_final_fit/round1000_c4_n150_multiseed/summary.json"
    )
    promoted_eval_complete = (
        root / "task_retention_promoted_final_eval/complete"
    ).is_file()
    if completion and completion.get("status") == "complete":
        current_stage = "complete"
    elif promoted_fit_complete or promoted_eval_complete:
        current_stage = "promoted_final_evaluation"
    elif promoted_fit_deferred:
        current_stage = "candidate_repair_diagnostics"
    elif promoted_experiment:
        current_stage = "promoted_exact_epoch_fit"
    else:
        current_stage = "task_retention_candidate_evaluation"
    experiments = {
        "legacy_exact_epoch_fit": {
            "experiment": legacy_experiment,
            "superseded": legacy_superseded,
        },
        "task_retention_rehearsal": {
            "experiment": text(root / "task_retention_rehearsal/active_experiment.txt"),
            "complete": rehearsal_complete,
        },
        "eedi_state_path_ab": {
            "experiment": text(root / "eedi_state_path_ab/active_experiment.txt"),
            "complete": (root / "eedi_state_path_ab/complete").is_file(),
            "role": "parallel_objective_diagnostic",
        },
        "state_kd": {
            "experiment": text(root / "state_kd/active_experiment.txt"),
            "complete": (root / "state_kd/complete").is_file(),
            "role": "closed_state_kd_without_direct_replay_diagnostic",
        },
        "equal_state_path": {
            "experiment": text(root / "equal_state_path/active_experiment.txt"),
            "complete": (root / "equal_state_path/complete").is_file(),
            "role": "equal_curriculum_objective_diagnostic",
        },
        "equal_state_kd": {
            "experiment": text(root / "equal_state_kd/active_experiment.txt"),
            "complete": (root / "equal_state_kd/complete").is_file(),
            "role": "equal_curriculum_state_kd_objective_diagnostic",
        },
        "equal_state_kd_open_t3": {
            "experiment": text(root / "equal_state_kd_open_t3/active_experiment.txt"),
            "complete": (root / "equal_state_kd_open_t3/complete").is_file(),
            "role": "equal_curriculum_open_t3_selective_kd_diagnostic",
        },
        "equal_state_path_open_t3": {
            "experiment": text(root / "equal_state_path_open_t3/active_experiment.txt"),
            "complete": (root / "equal_state_path_open_t3/complete").is_file(),
            "role": "equal_curriculum_open_t3_state_path_diagnostic",
        },
        "equal_state_kd_open_t3_lr_scan": {
            "experiment": text(
                root / "equal_state_kd_open_t3_lr_scan/active_experiment.txt"
            ),
            "complete": (
                root / "equal_state_kd_open_t3_lr_scan/complete"
            ).is_file(),
            "role": "equal_curriculum_open_t3_selective_kd_learning_rate_scan",
        },
        "equal_state_kd_open_t3_lr1e6": {
            "experiment": text(
                root / "equal_state_kd_open_t3_lr1e6/active_experiment.txt"
            ),
            "complete": (
                root / "equal_state_kd_open_t3_lr1e6/complete"
            ).is_file(),
            "role": "equal_curriculum_open_t3_selective_kd_lr1e6_midpoint",
        },
        "open_t3_consolidation": {
            "experiment": text(root / "open_t3_consolidation/active_experiment.txt"),
            "complete": (root / "open_t3_consolidation/complete").is_file(),
            "role": "two_stage_open_t3_causal_task_consolidation",
        },
        "staged_task_retention": {
            "experiment": text(root / "staged_task_retention/active_experiment.txt"),
            "complete": (root / "staged_task_retention/complete").is_file(),
            "role": "staged_curriculum_standard_diagnostic",
        },
        "staged_state_path": {
            "experiment": text(root / "staged_state_path/active_experiment.txt"),
            "complete": (root / "staged_state_path/complete").is_file(),
            "role": "staged_curriculum_state_path_diagnostic",
        },
        "eedi_full_lr_scan": {
            "experiment": text(root / "eedi_full_lr_scan/active_experiment.txt"),
            "complete": (root / "eedi_full_lr_scan/complete").is_file(),
            "role": "full_support_state_path_learning_rate_scan",
        },
        "eedi_full_state_kd_lr_scan": {
            "experiment": text(
                root / "eedi_full_state_kd_lr_scan/active_experiment.txt"
            ),
            "complete": (
                root / "eedi_full_state_kd_lr_scan/complete"
            ).is_file(),
            "role": "full_support_state_kd_learning_rate_scan",
        },
        "eedi_full_open_t3_lr_scan": {
            "experiment": text(
                root / "eedi_full_open_t3_lr_scan/active_experiment.txt"
            ),
            "complete": (
                root / "eedi_full_open_t3_lr_scan/complete"
            ).is_file(),
            "role": "full_support_open_t3_state_path_learning_rate_scan",
        },
        "eedi_full_open_t3_selective_kd_lr_scan": {
            "experiment": text(
                root
                / "eedi_full_open_t3_selective_kd_lr_scan/active_experiment.txt"
            ),
            "complete": (
                root / "eedi_full_open_t3_selective_kd_lr_scan/complete"
            ).is_file(),
            "role": "full_support_open_t3_selective_kd_learning_rate_scan",
        },
        "eedi_open_t3_t6_repair": {
            "experiment": text(
                root / "eedi_open_t3_t6_repair/active_experiment.txt"
            )
            or "layerwise-eedi-open-t3-t6-repair-seed42-20260803",
            "complete": (root / "eedi_open_t3_t6_repair/complete").is_file(),
            "scientific_gate_pass": (
                root / "eedi_open_t3_t6_repair/scientific_gate_pass"
            ).is_file(),
            "role": "parallel_open_t3_t6_repair_scan",
        },
        "external_state_teacher_eedi_full": {
            "experiment": text(
                root / "external_state_teacher_eedi_full_20260804/active_experiment.txt"
            ),
            "complete": (
                root / "external_state_teacher_eedi_full_20260804/complete"
            ).is_file(),
            "scientific_gate_pass": (
                root
                / "external_state_teacher_eedi_full_20260804/scientific_gate_pass"
            ).is_file(),
            "role": "gold_task_adapter_state_trajectory_repair_diagnostic",
        },
        "gold_semantic_teacher_eedi_full": {
            "experiment": text(
                root / "gold_semantic_teacher_eedi_full_20260804/active_experiment.txt"
            ),
            "complete": (
                root / "gold_semantic_teacher_eedi_full_20260804/complete"
            ).is_file(),
            "scientific_gate_pass": (
                root
                / "gold_semantic_teacher_eedi_full_20260804/scientific_gate_pass"
            ).is_file(),
            "role": "gold_diagnosis_semantic_state_prototype_repair_diagnostic",
        },
        "promoted_exact_epoch_fit": {
            "experiment": promoted_experiment,
            "complete": promoted_fit_complete,
            "deferred": promoted_fit_deferred,
        },
        "promoted_round1000_c4_n150_multiseed": {
            "experiment": "promoted-round1000-c4-n150-multiseed-20260804",
            "complete": (
                root
                / "task_retention_promoted_final_fit/round1000_c4_n150_multiseed/complete"
            ).is_file(),
            "scientific_gate_pass": bool(
                round1000_multiseed
                and round1000_multiseed.get("two_of_three_gate_pass") is True
            ),
            "role": "diagnostic_pair_seed_stability_not_final_gate",
        },
        "promoted_round1000_t3_semantic": {
            "experiment": None,
            "complete": (
                root
                / "task_retention_promoted_final_fit/local_round1000_t3_semantic/complete"
            ).is_file(),
            "scientific_gate_pass": (
                root
                / "task_retention_promoted_final_fit/local_round1000_t3_semantic/scientific_gate_pass"
            ).is_file(),
            "role": "diagnostic_T3_semantic_retention_not_final_gate",
        },
        "promoted_final_evaluation": {
            "experiment": text(
                root / "task_retention_promoted_final_eval/active_experiment.txt"
            ),
            "complete": promoted_eval_complete,
        },
    }
    result = {
        "stage": "layerwise_unified_live_status",
        "current_pipeline_stage": current_stage,
        "observed_at": now.isoformat(),
        "auth": {
            "feeds_monitoring_ready": not auth_required,
            "auth_required": auth_required,
            "remote_state_readable": not auth_required,
        },
        "training": {
            "experiment": active_experiment,
            "legacy_superseded_experiment": legacy_experiment
            if legacy_superseded
            else None,
            "last_progress_experiment": progress_experiment,
            "progress_matches_active_experiment": progress_matches_active,
            "last_observed_round": current_progress.get("round") if current_progress else None,
            "total_rounds": current_progress.get("total_rounds") if current_progress else None,
            "last_observed_status": current_progress.get("amlt_status")
            if current_progress
            else None,
            "last_remote_observation": observed.isoformat() if observed else None,
            "observation_age_seconds": progress_age,
            "observation_fresh": progress_fresh,
            "state_claim": (
                "awaiting_first_training_record"
                if awaiting_first_record and not auth_required
                else (
                    "fresh_remote_observation"
                    if progress_fresh and not auth_required
                    else (
                        "awaiting_current_experiment_observation"
                        if progress and not progress_matches_active
                        else "last_known_remote_observation_only"
                    )
                )
            ),
            "health_at_last_observation": health.get("healthy"),
            "health_source": health_source,
            "eta_at_last_observation": current_progress.get("eta_at")
            if current_progress
            else None,
        },
        "experiments": experiments,
        "controllers": controllers,
        "attention": attention,
        # Terminal owners remain visible for provenance, but their stale
        # heartbeats must not make the active pipeline look unhealthy.  A
        # nonterminal owner still fails closed if its heartbeat is stale.
        "required_controllers": sorted(
            name for name, row in controllers.items() if row["required_now"]
        ),
        "all_controllers_alive_locally": all(
            row["alive_locally"]
            for row in controllers.values()
            if row["required_now"]
        ),
        "overall_complete": bool(completion and completion.get("status") == "complete"),
        "completion_receipt": str(root / "overall_completion.json") if completion else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
