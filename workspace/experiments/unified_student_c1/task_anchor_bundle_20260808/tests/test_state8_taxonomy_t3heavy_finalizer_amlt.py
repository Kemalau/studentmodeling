from pathlib import Path

from scripts.validate_state8_taxonomy_t3heavy_finalizer_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_t3heavy_finalizer_is_single_gpu_validation_only_and_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_taxonomy_t3heavy_finalizer_20260807.yaml")
