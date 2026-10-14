from pathlib import Path

from scripts.validate_state8_matched_checkpoint2000_rescue_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint2000_rescue_config_is_hash_bound_and_compliant() -> None:
    validate(ROOT / "amlt/feeds_state8_matched_checkpoint2000_rescue_20260810.yaml")


def test_checkpoint2000_rescue_wrapper_has_required_scientific_controls() -> None:
    text = (ROOT / "scripts/run_state8_matched_checkpoint2000_rescue.sh").read_text()
    for fragment in (
        "STATE8_RANDOM_NEUTRAL=1",
        "STATE8_CAUSAL_REPEAT_FRACTION=0.50",
        "STATE8_CAUSAL_REPEAT_FRACTION=1.0",
        "STATE8_GEOMETRY_WEIGHT=0.10",
        "STATE8_GEOMETRY_WEIGHT=0.20",
        "STATE8_EEDI_GEOMETRY_WEIGHT=0.05",
        "STATE8_EEDI_GEOMETRY_WEIGHT=0.10",
        "STATE8_FORMAL_SCOPE=unified",
        "--num-pairs 50 --allow-manifest-prefix --seed 42",
        "semantic_judge_authoritative",
        "exact_match_authoritative",
        "gate['support']==75",
        "sum(x['support'] for x in gate['by_group'].values())==75",
        "all(x['pass'] is True for x in gate['by_group'].values())",
        "state8_diagnosis_first_full_history_teacher_screen_v1",
        "tg['support']==75",
        "tg['exact_match_authoritative'] is False",
    ):
        assert fragment in text
