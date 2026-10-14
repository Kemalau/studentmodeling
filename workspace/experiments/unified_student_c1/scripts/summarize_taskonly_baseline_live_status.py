#!/usr/bin/env python3
"""Build a fail-closed local status for the task-only baseline pipeline."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASKS = {
    "T1": "taskonly_qwen_t1_correctness_oneepoch_seed42_v1",
    "T2": "taskonly_qwen_t2_response_oneepoch_seed42_v1",
    "T3": "taskonly_qwen_t3_full_oneepoch_seed42_v1",
    "T4": "taskonly_qwen_t4_prospective_oneepoch_seed42_v1",
    "T5": "taskonly_qwen_t5_persistence_oneepoch_seed42_v1",
}

REMOTE_JOBS = {
    "T1": "qwen3-32b-taskonly-t1-correctness-oneepoch-seed42",
    "T2": "qwen3-32b-taskonly-t2-response-oneepoch-seed42",
    "T3": "qwen3-32b-taskonly-t3-full-oneepoch-seed42",
    "T4": "qwen3-32b-taskonly-t4-prospective-oneepoch-seed42",
    "T5": "qwen3-32b-taskonly-t5-persistence-oneepoch-seed42",
}

REMOTE_JOB_IDS = {
    "T1": "careful_toe_p87sfl2prk",
    "T2": "strong_pepper_8ft9yhwng1",
    "T3": "lemon_button_fyw8ljlmqx",
    "T4": "bubbly_carpet_7wbmd8wxx1",
    "T5": "bright_yam_cpt3f61t59",
}


def read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8", errors="replace").strip()
    return value or None


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def process_alive(path: Path) -> bool:
    value = read_text(path)
    if value is None or not value.isdigit():
        return False
    try:
        os.kill(int(value), 0)
    except (OSError, ValueError):
        return False
    return True


def status_value(path: Path) -> str | None:
    value = read_text(path)
    if value is None:
        return None
    fields = value.split("\t")
    return fields[-1] if fields else None


def remote_status(rows: Any, job: str) -> str | None:
    if not isinstance(rows, list):
        return None
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("job_name") or "").lstrip(":") == job
    ]
    if len(matches) != 1:
        return None
    value = str(matches[0].get("status") or "").strip().casefold()
    return value or None


def summarize(project: Path, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    result_root = project / "results/taskonly_qwen_formal_eval_20260813"
    supervisor_root = project / "results/taskonly_baseline_supervisor_20260813"
    train_roots = {
        "T1": project / "results/taskonly_qwen_long_train_20260813/taskonly_t1",
        "T2": project / "results/taskonly_qwen_long_train_20260813/taskonly_t2",
        "T3": project / "results/taskonly_qwen_short_train_20260813/taskonly_t3",
        "T4": project / "results/taskonly_qwen_long_train_20260813/taskonly_t4",
        "T5": project / "results/taskonly_qwen_short_train_20260813/taskonly_t5",
    }
    recovery_roots = {
        task: project / f"results/taskonly_qwen_resumable_recovery_20260814/{task}"
        for task in ("T1", "T2", "T4")
    }
    long_remote = read_json(supervisor_root / "train_long_status.json")
    short_remote = read_json(supervisor_root / "train_short_status.json")
    tasks: dict[str, Any] = {}
    for task, adapter_tag in TASKS.items():
        train = train_roots[task]
        formal = result_root / task
        remote = remote_status(
            short_remote if task in {"T3", "T5"} else long_remote,
            REMOTE_JOBS[task],
        )
        recovery = recovery_roots.get(task)
        recovery_status = status_value(recovery / "latest_status.tsv") if recovery else None
        recovery_experiment = read_text(recovery / "active_experiment.txt") if recovery else None
        effective_status = recovery_status or remote or status_value(train / "latest_status.tsv")
        tasks[task] = {
            "adapter_tag": adapter_tag,
            "remote_job": REMOTE_JOBS[task],
            "remote_job_id": REMOTE_JOB_IDS[task],
            "training_status": remote or status_value(train / "latest_status.tsv"),
            "effective_training_status": effective_status,
            "training_status_source": "amlt_experiment_snapshot" if remote else "local_watcher",
            "training_watcher_alive": process_alive(train / "watcher.active.pid"),
            "remote_passed_marker": (train / "remote_passed").is_file(),
            "remote_failed_marker": (train / "remote_failed").is_file(),
            "adapter_receipt_validated": (formal / "adapter_ready").is_file(),
            "formal_controller_alive": process_alive(formal / "controller.active.pid"),
            "formal_status": status_value(formal / "latest_status.tsv"),
            "formal_complete": (formal / "complete").is_file(),
            "needs_attention": (formal / "needs_attention").is_file(),
            "recovery_status": recovery_status,
            "recovery_experiment": recovery_experiment,
            "recovery_controller_alive": bool(
                recovery and process_alive(recovery / "controller.active.pid")
            ),
            "recovery_needs_attention": bool(
                recovery and (recovery / "needs_attention").is_file()
            ),
        }

    t3_eedi_root = project / "results/t3_eedi_taskonly_qwen_formal_20260813"
    t3_eedi_metrics = read_json(t3_eedi_root / "validated/metrics.json")
    t3_group = None
    if isinstance(t3_eedi_metrics, dict):
        t3_group = (t3_eedi_metrics.get("metrics_by_group") or {}).get(
            "diagnose_misconception/eedi_mining/open_taxonomy_generation"
        )
    t3_eedi_complete = bool(
        (t3_eedi_root / "complete").is_file()
        and isinstance(t3_group, dict)
        and int(t3_group.get("n", -1)) == 894
        and int(t3_group.get("llm_judge_n", -1)) == 894
    )
    unified_t3_v2_root = project / "results/formal_specialist_v2/unrestricted_qwen_t3_v2"
    unified_t3_v2_receipt = read_json(unified_t3_v2_root / "completion.json")
    unified_t3_v2_complete = bool(
        (unified_t3_v2_root / "complete").is_file()
        and isinstance(unified_t3_v2_receipt, dict)
        and unified_t3_v2_receipt.get("status") == "complete"
        and int(unified_t3_v2_receipt.get("base_n", -1)) == 1494
        and int(unified_t3_v2_receipt.get("unified_n", -1)) == 1494
    )
    summary = read_json(result_root / "summary/summary.json")
    matrix = read_json(result_root / "formal_specialist_v3/formal_evaluation_matrix.json")
    exposure_audit = read_json(result_root / "target_exposure_audit.json")
    exposure_match = bool(
        isinstance(exposure_audit, dict)
        and exposure_audit.get("status") == "complete"
        and exposure_audit.get("target_task_exposure_exact_match") is True
        and int(exposure_audit.get("total_unified_rows", -1)) == 259_845
        and all(
            isinstance(row, dict)
            and row.get("id_set_equal") is True
            and row.get("target_supervision_equal") is True
            and int(row.get("target_signature_mismatches", -1)) == 0
            for row in (exposure_audit.get("tasks") or {}).values()
        )
        and len(exposure_audit.get("tasks") or {}) == 5
    )
    taskonly_complete = bool(
        (result_root / "complete").is_file()
        and isinstance(summary, dict)
        and summary.get("status") == "complete"
        and isinstance(matrix, dict)
    )
    auth_ready = (supervisor_root / "auth_ready").is_file()
    all_complete = taskonly_complete and t3_eedi_complete
    return {
        "stage": "taskonly_baseline_live_status_v1",
        "observed_at": now.isoformat(),
        "feeds_monitoring_ready": auth_ready,
        "tasks": tasks,
        "target_task_exposure": {
            "complete": exposure_match,
            "receipt": exposure_audit if exposure_match else None,
        },
        "taskonly_matrix_complete": taskonly_complete,
        "taskonly_summary": summary,
        "unrestricted_t3_v2_rejudge": {
            "complete": unified_t3_v2_complete,
            "controller_alive": process_alive(unified_t3_v2_root / "controller.active.pid"),
            "needs_attention": (unified_t3_v2_root / "needs_attention").is_file(),
            "receipt": unified_t3_v2_receipt,
        },
        "t3_eedi": {
            "experiment": "t3-eedi-taskonly-qwen-formal-recovery1-20260813",
            "job": "qwen3-32b-t3-eedi-taskonly-formal-open-seed42",
            "status": status_value(t3_eedi_root / "latest_status.tsv"),
            "complete": t3_eedi_complete,
            "metrics": t3_group,
            "needs_attention": (t3_eedi_root / "needs_attention").is_file(),
        },
        "needs_attention": sorted(
            [task for task, value in tasks.items() if value["needs_attention"]]
            + [
                f"{task}-recovery"
                for task, value in tasks.items()
                if value["recovery_needs_attention"]
            ]
            + (["T3-Eedi"] if (t3_eedi_root / "needs_attention").is_file() else [])
            + (["Unified-T3-v2"] if (unified_t3_v2_root / "needs_attention").is_file() else [])
        ),
        "overall_complete": all_complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = summarize(args.project)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
