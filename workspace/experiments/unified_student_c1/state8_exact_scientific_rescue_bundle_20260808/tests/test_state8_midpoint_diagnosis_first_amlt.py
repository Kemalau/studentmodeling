from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_midpoint_diagnosis_first_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_midpoint_diagnosis_first_repair_20260808.yaml"
BUNDLE = ROOT / "state8_midpoint_diagnosis_first_bundle_20260808"


def test_midpoint_diagnosis_first_config_is_hash_bound_and_resumable() -> None:
    validate(CONFIG, BUNDLE)


def test_validator_rejects_midpoint_source_hash_drift(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value = copy.deepcopy(value)
    value["jobs"][0]["submit_args"]["env"]["EXPECTED_STUDENT_STATE_SHA256"] = "0" * 64
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError, match="Environment drift"):
        validate(path, BUNDLE)
