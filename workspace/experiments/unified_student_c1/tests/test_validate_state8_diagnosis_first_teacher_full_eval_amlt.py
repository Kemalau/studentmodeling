from pathlib import Path

from scripts.validate_state8_diagnosis_first_teacher_full_eval_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_full_diagnosis_teacher_eval_config() -> None:
    validate(ROOT / "amlt/feeds_state8_diagnosis_first_teacher_full_eval_16way_20260810.yaml")
