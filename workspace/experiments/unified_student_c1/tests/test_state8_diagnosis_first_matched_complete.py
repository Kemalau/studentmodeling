from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from materialize_state8_diagnosis_first_matched_final_eval import (  # noqa: E402
    CANONICAL,
    PROGRESS,
    materialize,
)
from validate_state8_final_eval_amlt import load, validate  # noqa: E402


def test_materialized_final_eval_is_bound_and_valid(tmp_path: Path) -> None:
    output = tmp_path / "eval.yaml"
    materialize(ROOT / "amlt/feeds_state8_final_eval_20260805.yaml", output)
    config = load(output)
    assert len(config["jobs"]) == 19
    for job in config["jobs"]:
        env = job["submit_args"]["env"]
        assert env["STATE8_FINAL_EVAL_RUN_DIR"] == CANONICAL
        assert env["STATE8_FINAL_EVAL_PROGRESS_NAMESPACE"] == PROGRESS
    validate(config)


def test_complete_controller_is_fail_closed_and_resumable() -> None:
    source = (ROOT / "scripts/watch_state8_diagnosis_first_matched_complete.sh").read_text()
    for fragment in (
        "state8_diagnosis_first_matched_joint_promotion_v1",
        "state8-diagnosis-first-matched-seed67-unified-finalfit-seed42",
        "fit_validator",
        "count_active_a100_requests.py",
        "available_slots",
        "quota_slots",
        "amlt resume",
        "wave=$((wave+1))",
        "259845",
        "diagnosis_evidence_label_v1",
        "watch_state8_final_eval.sh",
        "validate_complete_unified_objective.py",
        "monitor_layerwise_unified_training_progress.py",
        "STATE8_MATCHED_FINALFIT_STALL_SECONDS:-7200",
        "STATE8_MATCHED_FINALFIT_STARTUP_STALL_SECONDS:-14400",
        "STATE8_MATCHED_FINALFIT_POSTTRAIN_STALL_SECONDS:-21600",
        "select_state8_finalfit_topology.py",
        "--profile diagnosis_first_matched",
        "fit_distributed_rounds",
        "fit_gpu_count",
        "cross_topology_resume_allowed",
        'amlt logs view -F "$log_file"',
        'amlt cancel "$experiment" ":$job" -y',
        "stall_cancel_requested",
        "cancelled stalled/unhealthy matched final fit for atomic-checkpoint recovery",
    ):
        assert fragment in source
