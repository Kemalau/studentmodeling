from pathlib import Path


SOURCE = Path("scripts/train_state8_eedi_same_question_finalfit.py").read_text()


def test_same_question_finalfit_binds_all_required_contracts():
    for fragment in (
        "state8_diagnosis_first_teacher_full_provenance_gate_v1",
        "diagnosis_evidence_label_v1",
        "STATE8_MATCHED_FULL_TEACHER_GATE_SHA256",
        "STATE8_MATCHED_FULL_TEACHER_PROVENANCE_SHA256",
        '"teacher_gate_support": 1494',
        '"history_dropout_schedule": [1.0]',
        '"raw_history_visible_to_decoder": False',
        "current_item_visible_response_state_only_v1:eedi_only",
        '"student_response_visible_to_decoder": False',
        '"same_question_causal_replay_required": True',
        '"semantic_judge_authoritative": True',
        '"exact_match_authoritative": False',
    ):
        assert fragment in SOURCE
