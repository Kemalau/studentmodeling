from pathlib import Path

from scripts.validate_taskonly_qwen_short_train_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_taskonly_short_train_jobs_are_one_epoch_and_remote() -> None:
    validate(ROOT / "amlt/feeds_taskonly_qwen_short_train_20260813.yaml")
