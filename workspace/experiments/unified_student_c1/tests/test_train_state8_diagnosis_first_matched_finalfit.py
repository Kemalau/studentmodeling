from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "scripts/train_state8_diagnosis_first_matched_finalfit.py").read_text()


def test_matched_finalfit_binds_full_teacher_and_state_only_schema() -> None:
    for fragment in (
        "state8_diagnosis_first_teacher_full_provenance_gate_v1",
        "diagnosis_evidence_label_v1", "STATE8_MATCHED_FULL_TEACHER_GATE_SHA256",
        "STATE8_MATCHED_FULL_TEACHER_PROVENANCE_SHA256", '"teacher_gate_support": 1494',
        '"history_dropout_schedule": [1.0]', '"raw_history_visible_to_decoder": False',
        '["diagnosis", "evidence", "label"]', "auxiliary_ordinary_weight",
        '"semantic_judge_authoritative": True', '"exact_match_authoritative": False',
    ):
        assert fragment in SOURCE


def test_canonical_publisher_knows_matched_variant() -> None:
    publisher = (ROOT / "scripts/publish_state8_final_checkpoint.py").read_text()
    assert '"diagnosis_first_matched_seed67"' in publisher
    assert 'Path("diagnosis_first_matched_finalfit.json")' in publisher
