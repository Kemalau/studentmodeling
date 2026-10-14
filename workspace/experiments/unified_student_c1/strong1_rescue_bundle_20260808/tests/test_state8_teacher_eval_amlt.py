from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_state8_teacher_eval_amlt import load, validate  # noqa: E402


def test_teacher_eval_config_passes() -> None:
    validate(load(ROOT / "amlt/feeds_state8_teacher_eval_20260805.yaml"))
