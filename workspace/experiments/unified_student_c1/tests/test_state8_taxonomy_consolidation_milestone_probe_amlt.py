from pathlib import Path

from scripts.validate_state8_taxonomy_consolidation_milestone_probe_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_consolidation_milestone_probe_is_fail_closed() -> None:
    validate(ROOT / "amlt/feeds_state8_taxonomy_consolidation_milestone_probe_20260808.yaml")
