from pathlib import Path

from scripts.validate_state8_diagnosis_first_teacher_checkpoint200_screen_amlt import validate


ROOT=Path(__file__).resolve().parents[1]


def test_checkpoint200_screen_config_is_frozen() -> None:
    validate(ROOT/'amlt/feeds_state8_diagnosis_first_teacher_checkpoint200_screen_20260810.yaml')
