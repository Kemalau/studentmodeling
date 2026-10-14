from pathlib import Path

from scripts.validate_t3_eedi_taskonly_qwen_formal_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_t3_eedi_taskonly_formal_config_is_hash_bound_and_remote() -> None:
    validate(ROOT / "amlt/feeds_t3_eedi_taskonly_qwen_formal_20260813.yaml")
