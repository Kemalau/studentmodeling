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


def test_consolidation_saves_every_five_steps_but_recovers_any_atomic_step() -> None:
    source = (ROOT / "scripts/run_state8_stateonly_repair_ab.sh").read_text()
    assert 'STATE8_REPAIR_SAVE_STEPS="${STATE8_REPAIR_SAVE_STEPS:-1}"' in source
    assert 'STATE8_REPAIR_RECOVERY_SAVE_STEPS="${STATE8_REPAIR_RECOVERY_SAVE_STEPS:-$STATE8_REPAIR_SAVE_STEPS}"' in source
    assert '--save-steps "$STATE8_REPAIR_SAVE_STEPS"' in source
    assert "--expected-planned-optimizer-steps 250" in source
    assert '--recovery-save-steps "$STATE8_REPAIR_RECOVERY_SAVE_STEPS"' in source
    config = (
        ROOT / "amlt/feeds_state8_taxonomy_curriculum_consolidation_20260807.yaml"
    ).read_text()
    assert "STATE8_REPAIR_SAVE_STEPS: '5'" in config
    assert "STATE8_REPAIR_RECOVERY_SAVE_STEPS: '1'" in config
