from pathlib import Path

from scripts.validate_state8_matched_checkpoint1200_diagnosis_c4_shards_amlt import load, validate


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint1200_c4_shards_are_disjoint_and_hash_bound() -> None:
    config = ROOT / "amlt/feeds_state8_matched_checkpoint1200_diagnosis_c4_shards_20260810.yaml"
    validate(load(config), source_root=ROOT)
