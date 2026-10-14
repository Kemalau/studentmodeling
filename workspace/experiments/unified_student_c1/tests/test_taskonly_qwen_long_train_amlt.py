from pathlib import Path

from scripts.validate_taskonly_qwen_long_train_amlt import validate


ROOT=Path(__file__).resolve().parents[1]


def test_taskonly_long_jobs_are_hash_bound_one_epoch_controls() -> None:
    validate(ROOT / "amlt/feeds_taskonly_qwen_long_train_20260813.yaml")
