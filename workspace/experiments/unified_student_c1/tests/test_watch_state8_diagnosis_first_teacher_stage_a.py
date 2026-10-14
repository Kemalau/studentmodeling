from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/watch_state8_diagnosis_first_teacher_stage_a.sh").read_text()


def test_teacher_watcher_is_semantic_gated_and_resumable() -> None:
    for fragment in (
        "--judge-backend trapi",
        "semantic_teacher_gate_pass",
        "submit_recovery",
        "feeds_submission_global.lock",
        "max_recovery_waves=12",
        "active_experiment.txt",
        "validate_state8_diagnosis_first_teacher_stage_a_amlt.py",
        "validate_state8_diagnosis_first_matched_curriculum_amlt.py",
    ):
        assert fragment in SOURCE
