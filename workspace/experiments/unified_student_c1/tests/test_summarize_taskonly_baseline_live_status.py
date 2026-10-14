from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_taskonly_baseline_live_status import summarize  # noqa: E402


def test_live_status_fails_closed_without_artifacts(tmp_path: Path) -> None:
    value = summarize(tmp_path)
    assert value["overall_complete"] is False
    assert value["taskonly_matrix_complete"] is False
    assert value["t3_eedi"]["complete"] is False
    assert value["unrestricted_t3_v2_rejudge"]["complete"] is False
    assert value["target_task_exposure"]["complete"] is False
    assert set(value["tasks"]) == {"T1", "T2", "T3", "T4", "T5"}
    assert value["tasks"]["T1"]["remote_job_id"] == "careful_toe_p87sfl2prk"
    assert value["tasks"]["T5"]["remote_job_id"] == "bright_yam_cpt3f61t59"


def test_t3_eedi_requires_full_judged_support(tmp_path: Path) -> None:
    root = tmp_path / "results/t3_eedi_taskonly_qwen_formal_20260813"
    (root / "validated").mkdir(parents=True)
    (root / "complete").touch()
    metrics = {
        "metrics_by_group": {
            "diagnose_misconception/eedi_mining/open_taxonomy_generation": {
                "n": 894,
                "llm_judge_n": 893,
            }
        }
    }
    (root / "validated/metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert summarize(tmp_path)["t3_eedi"]["complete"] is False
    metrics["metrics_by_group"]["diagnose_misconception/eedi_mining/open_taxonomy_generation"]["llm_judge_n"] = 894
    (root / "validated/metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert summarize(tmp_path)["t3_eedi"]["complete"] is True


def test_remote_experiment_snapshot_overrides_stale_watcher_status(tmp_path: Path) -> None:
    supervisor = tmp_path / "results/taskonly_baseline_supervisor_20260813"
    supervisor.mkdir(parents=True)
    rows = [
        {
            "job_name": ":qwen3-32b-taskonly-t1-correctness-oneepoch-seed42",
            "status": "running",
        },
        {
            "job_name": ":qwen3-32b-taskonly-t2-response-oneepoch-seed42",
            "status": "queued",
        },
        {
            "job_name": ":qwen3-32b-taskonly-t4-prospective-oneepoch-seed42",
            "status": "pass",
        },
    ]
    (supervisor / "train_long_status.json").write_text(json.dumps(rows), encoding="utf-8")
    value = summarize(tmp_path)
    assert value["tasks"]["T1"]["training_status"] == "running"
    assert value["tasks"]["T2"]["training_status"] == "queued"
    assert value["tasks"]["T4"]["training_status"] == "pass"
    assert value["tasks"]["T4"]["training_status_source"] == "amlt_experiment_snapshot"


def test_recovery_status_is_reported_without_hiding_original_terminal_state(tmp_path: Path) -> None:
    supervisor = tmp_path / "results/taskonly_baseline_supervisor_20260813"
    supervisor.mkdir(parents=True)
    (supervisor / "train_long_status.json").write_text(
        json.dumps(
            [
                {
                    "job_name": ":qwen3-32b-taskonly-t1-correctness-oneepoch-seed42",
                    "status": "killed",
                }
            ]
        ),
        encoding="utf-8",
    )
    recovery = tmp_path / "results/taskonly_qwen_resumable_recovery_20260814/T1"
    recovery.mkdir(parents=True)
    (recovery / "latest_status.tsv").write_text(
        "2026-08-14T00:00:00Z\twaiting_for_user_a100_slot\n", encoding="utf-8"
    )
    (recovery / "active_experiment.txt").write_text(
        "taskonly-qwen-t1-resumable-recovery1-20260814\n", encoding="utf-8"
    )
    value = summarize(tmp_path)
    assert value["tasks"]["T1"]["training_status"] == "killed"
    assert value["tasks"]["T1"]["effective_training_status"] == "waiting_for_user_a100_slot"
    assert value["tasks"]["T1"]["recovery_experiment"].endswith("recovery1-20260814")


def test_target_exposure_requires_ids_targets_and_all_five_tasks(tmp_path: Path) -> None:
    root = tmp_path / "results/taskonly_qwen_formal_eval_20260813"
    root.mkdir(parents=True)
    tasks = {
        task: {
            "id_set_equal": True,
            "target_supervision_equal": True,
            "target_signature_mismatches": 0,
        }
        for task in ("T1", "T2", "T3", "T4", "T5")
    }
    receipt = {
        "status": "complete",
        "target_task_exposure_exact_match": True,
        "total_unified_rows": 259_845,
        "tasks": tasks,
    }
    (root / "target_exposure_audit.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert summarize(tmp_path)["target_task_exposure"]["complete"] is True
    receipt["tasks"]["T3"]["target_signature_mismatches"] = 1
    (root / "target_exposure_audit.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert summarize(tmp_path)["target_task_exposure"]["complete"] is False
