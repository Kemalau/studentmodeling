from pathlib import Path

from scripts.validate_state8_matched_checkpoint3000_semantic_finalizer_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_finalizer_config_is_hash_bound_and_valid() -> None:
    validate(ROOT / "amlt/feeds_state8_matched_checkpoint3000_semantic_finalizer_20260811.yaml")


def test_finalizer_binds_fix_and_completed_checkpoint() -> None:
    text = (ROOT / "scripts/run_state8_matched_checkpoint3000_semantic_finalizer.sh").read_text()
    for fragment in (
        "7d5e4966a5317733fe5d5f32cc6730eb82f4ad78dedd8fa6a6967ee4aefb4e6d",
        "checkpoint-3000",
        "receipt['step']==3000",
        "export TRAIN_SEED=3066",
        "semantic_preserve",
    ):
        assert fragment in text
