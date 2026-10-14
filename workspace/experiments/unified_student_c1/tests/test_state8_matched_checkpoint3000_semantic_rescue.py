from pathlib import Path

from scripts.validate_state8_matched_checkpoint3000_semantic_rescue_amlt import validate


ROOT=Path(__file__).resolve().parents[1]


def test_cp3000_rescue_config_is_hash_bound_and_valid() -> None:
    validate(ROOT/"amlt/feeds_state8_matched_checkpoint3000_semantic_rescue_20260810.yaml")


def test_cp3000_runner_changes_only_the_bound_source_and_seed() -> None:
    text=(ROOT/"scripts/run_state8_matched_checkpoint3000_semantic_rescue.sh").read_text()
    for fragment in ("checkpoint-3000","receipt['step']==3000","si.get('step')==3000","export TRAIN_SEED=3066","semantic_preserve"):
        assert fragment in text
