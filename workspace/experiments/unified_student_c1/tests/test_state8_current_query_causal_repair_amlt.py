from pathlib import Path

import yaml

from scripts.validate_state8_current_query_causal_repair_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_current_query_causal_repair_20260811.yaml"
EEDI_CONFIG = ROOT / "amlt/feeds_state8_eedi_current_query_causal_repair_20260811.yaml"
EEDI_SAME_QUESTION_CONFIG = (
    ROOT / "amlt/feeds_state8_eedi_same_question_causal_repair_20260811.yaml"
)
EEDI_SAME_QUESTION_FULL_CONFIG = (
    ROOT / "amlt/feeds_state8_eedi_same_question_full_mse1_20260811.yaml"
)
EEDI_SAME_QUESTION_FULL_MSE2_CONFIG = (
    ROOT / "amlt/feeds_state8_eedi_same_question_full_mse2_20260811.yaml"
)


def test_current_query_repair_config_is_valid() -> None:
    validate(CONFIG)
    validate(EEDI_CONFIG)
    validate(EEDI_SAME_QUESTION_CONFIG)
    validate(EEDI_SAME_QUESTION_FULL_CONFIG)
    validate(EEDI_SAME_QUESTION_FULL_MSE2_CONFIG)


def test_validator_rejects_history_route_drift(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["jobs"][0]["submit_args"]["env"]["SOURCE_ANCHOR_MODE"] = "full_mse1"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    try:
        validate(bad)
    except ValueError as error:
        assert "Environment" in str(error)
    else:
        raise AssertionError("Wrong repair arm was accepted")
