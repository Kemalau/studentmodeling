from pathlib import Path

from scripts.validate_state8_taxonomy_consolidation_round35_t3_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_round35_t3_probe_config_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_taxonomy_consolidation_round35_t3_20260808.yaml")
