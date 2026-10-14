from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/run_state8_diagnosis_first_matched_curriculum.sh").read_text()


def test_matched_curriculum_is_fail_closed_and_state_only() -> None:
    for fragment in (
        "state8_diagnosis_first_full_history_teacher_v1",
        "state8_diagnosis_first_full_history_teacher_screen_v1",
        "semantic_teacher_gate_pass",
        "STATE8_MATCHED_TEACHER_GATE_SHA256",
        "train_state8_diagnosis_first_matched_teacher_curriculum.py",
        "STATE8_SINGLE_DEVICE_FALLBACK=1",
        "STATE8_RUN_SUFFIX=diagnosis_first_matched_teacher_seed67",
        "['diagnosis','evidence','label']",
        "raw_history_visible_to_decoder",
        "evaluate_recurrent_diagnosis_c4_diagnosis_first.py",
        "--num-pairs 20",
        "semantic_judgment_pending",
        "evaluate_recurrent_causal_gate.py",
        "--num-pairs 150",
        "recurrent_causal_gate.json",
    ):
        assert fragment in SOURCE
