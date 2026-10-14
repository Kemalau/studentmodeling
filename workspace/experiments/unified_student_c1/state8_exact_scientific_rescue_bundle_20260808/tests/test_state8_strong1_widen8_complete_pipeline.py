from pathlib import Path

import yaml

from scripts.materialize_state8_strong1_final_eval_config import (
    CANONICAL,
    PROGRESS,
    materialize,
)
from scripts.validate_state8_final_eval_amlt import validate as validate_eval


ROOT = Path(__file__).resolve().parents[1]


def test_strong1_final_eval_derives_all_approved_jobs(tmp_path: Path) -> None:
    output = tmp_path / "amlt" / "generated.yaml"
    output.parent.mkdir()
    materialize(ROOT / "amlt/feeds_state8_final_eval_20260805.yaml", output)
    config = yaml.safe_load(output.read_text())
    assert len(config["jobs"]) == 19
    for job in config["jobs"]:
        env = job["submit_args"]["env"]
        assert env["STATE8_FINAL_EVAL_RUN_DIR"] == CANONICAL
        assert env["STATE8_FINAL_EVAL_PROGRESS_NAMESPACE"] == PROGRESS
    validate_eval(config)


def test_complete_supervisor_covers_selection_fit_recovery_and_final_eval() -> None:
    source = (ROOT / "scripts/supervise_state8_strong1_widen8_complete.sh").read_text()
    for fragment in (
        "state8_strong1_widen8_joint_promotion_v1",
        "state8_midpoint_diagnosis_first_joint_promotion_v1",
        "midpoint_diagnosis_first_seed63",
        "strong1_midpoint_diagnosis_first_seed63",
        "state8_midpoint_diagnosis_first_v1/selection.json",
        "state8_exact_diagnosis_curriculum_joint_promotion_v1",
        "exact_diagnosis_curriculum_seed65",
        "strong1_exact_diagnosis_curriculum_seed65",
        "state8_exact_diagnosis_curriculum_v1/selection.json",
        "select_state8_complete_candidate.py",
        "highest-scoring qualified candidate",
        "answer_C4_pass",
        "diagnosis_C4_pass",
        "feeds_state8_strong1_widen8_final_fit_20260808.yaml",
        "feeds_state8_strong1_widen8_final_fit_world2_20260808.yaml",
        "count_active_a100_requests.py",
        "select_state8_finalfit_topology.py",
        "topology_selection.json",
        "cross_topology_resume_allowed",
        'fit_validator="$project/${topology_fields[2]}"',
        'complete-finalfit-${fit_topology}-20260808',
        'recovery${wave}-${fit_topology}-20260808',
        "259845-row strict-state-only epoch",
        "materialize_state8_strong1_final_eval_config.py",
        "watch_state8_final_eval.sh",
        "scientific_gate_pass",
        "scientific_gate_failed",
        "recoverable final-evaluation controller exit",
        "retire_queued_experiment",
        'amlt cancel --status "$phase" --yes',
        "never kill a healthy running candidate",
        "redundant_candidates_retired",
        "superseded_before_run",
        "retired_for_diagnosis_first_priority",
        '|| -f "$rescue/not_required"',
        "heartbeat_pid",
        "while true; do heartbeat; sleep 30; done",
        'sleep_unlocked(){ sleep "$1" 9>&-; }',
    ):
        assert fragment in source

    wrapper = (
        ROOT / "scripts/watch_state8_strong1_widen8_complete_supervisor.sh"
    ).read_text()
    assert "supervise_state8_strong1_widen8_complete.sh" in wrapper
    assert "recoverable complete-supervisor exit" in wrapper


def test_short_candidate_controllers_do_not_recover_superseded_work() -> None:
    primary = (ROOT / "scripts/watch_state8_strong1_widen8_bridge.sh").read_text()
    fallback = (
        ROOT / "scripts/watch_state8_strong1_widen8_single_fallback.sh"
    ).read_text()
    assert "complete_pipeline/selection.json" in primary
    assert "skipping recovery after terminal state" in primary
    assert "skipping evaluation" in primary
    assert "complete_pipeline/selection.json" in fallback
    assert "retired_for_diagnosis_first_priority" in fallback
    assert "skipping recovery after terminal state" in fallback
    assert "skipping evaluation" in fallback
    assert "heartbeat_pid" in primary
    assert "heartbeat_pid" in fallback


def test_legacy_rescue_yields_to_exact_diagnosis_branch() -> None:
    source = (ROOT / "scripts/supervise_state8_strong1_rescue.sh").read_text()
    assert '"$bridge/superseded"' in source
    assert '"$fallback/retired_for_diagnosis_first_priority"' in source
    assert '"$exact_diagnosis/active_experiment.txt"' in source
    assert 'touch "$state/not_required" "$state/complete"' in source
    assert 'sleep_unlocked(){ sleep "$1" 9>&-; }' in source
