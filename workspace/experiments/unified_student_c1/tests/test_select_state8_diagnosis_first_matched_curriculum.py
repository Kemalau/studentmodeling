from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/select_state8_diagnosis_first_matched_curriculum.py").read_text()


def test_matched_selection_covers_the_full_joint_gate() -> None:
    for fragment in (
        "full_teacher_gate", "support\") != 1494", "evaluation_shards\") != 16",
        '"T1"', '"T2"', '"T3_eedi_semantic"', '"T5"', '"T6_macro_f1"',
        '"answer_C4_pass"', '"diagnosis_C4_pass"', "strict_state_only",
        "answer_c4_150/recurrent_causal_gate.json", "--full-teacher-provenance",
        "state8_diagnosis_first_matched_joint_promotion_v1",
    ):
        assert fragment in SOURCE
