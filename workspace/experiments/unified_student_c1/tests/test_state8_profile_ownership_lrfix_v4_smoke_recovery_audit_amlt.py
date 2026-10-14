from pathlib import Path

import pytest
import yaml

from scripts.validate_state8_profile_ownership_lrfix_v4_smoke_recovery_audit_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_profile_ownership_lrfix_v4_smoke_recovery_audit_20260821.yaml"


def test_recovery_config_is_frozen_and_valid() -> None:
    validate(CONFIG)


def test_recovery_config_rejects_evaluator_hash_drift(tmp_path: Path) -> None:
    value = yaml.safe_load(CONFIG.read_text())
    value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
        "d1297ba5c841511061ca6d40f2b4c4e2b431b0d6c91eadb2756c95b5f40f45e6",
        "0" * 64,
    )
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="source binding drift"):
        validate(path)
