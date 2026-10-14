from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from scripts.validate_state8_profile_ownership_checkpoint_audit_amlt import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_state8_profile_ownership_cp200_early_audit_20260820.yaml"
RUNNER = ROOT / "scripts/run_state8_profile_ownership_checkpoint_audit.sh"


def test_complete_config_passes() -> None:
    validate(CONFIG)


def test_config_is_bound_to_runner_hash() -> None:
    digest = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    assert digest in CONFIG.read_text(encoding="utf-8")


def test_runner_is_read_only_and_checkpoint_bound() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "checkpoint-200" in source
    assert "9f7f8e82f6dd5cc80522ebfd99ac972aa45b60234251c5f69af6f94c7d8ede58" in source
    assert "fc07a859f74ad5addbdf6bd4fde86ae3cfbc5063eb822323eafab097012cf7ba" in source
    assert "early_audit_state8_profile_ownership_checkpoint.py" in source
    assert not re.search(r"optimizer\.step|--init-adapter|train_state8", source)


def test_unresolved_runner_hash_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text(CONFIG.read_text().replace(
        hashlib.sha256(RUNNER.read_bytes()).hexdigest(), "RUNNER_SHA256"
    ))
    with pytest.raises(ValueError, match="placeholder"):
        validate(broken)
