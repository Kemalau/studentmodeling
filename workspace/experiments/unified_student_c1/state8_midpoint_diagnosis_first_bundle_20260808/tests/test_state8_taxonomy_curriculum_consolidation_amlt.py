from pathlib import Path

from scripts.validate_state8_taxonomy_curriculum_consolidation_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_taxonomy_curriculum_consolidation_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_taxonomy_curriculum_consolidation_20260807.yaml")


def test_consolidation_is_state_only_but_keeps_unified_screen() -> None:
    source = (ROOT / "scripts/run_state8_stateonly_repair_ab.sh").read_text()
    assert "STATE8_FORMAL_SCOPE" in source
    assert "FORMAL_SCOPE_ARGS" in source
    assert "--max-samples-per-group 25" in source
    assert "history_dropout_schedule = [1.0]" in (
        ROOT / "scripts/train_state8_stateonly_teacher_repair.py"
    ).read_text()
