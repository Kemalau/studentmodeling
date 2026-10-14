from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_audit_is_remote_eval_only_and_fail_closed() -> None:
    source = (
        ROOT / "scripts/run_state8_profile_ownership_checkpoint_audit.sh"
    ).read_text()
    assert "checkpoint_snapshots/state8_future_profile_ownership_seed8801_v3" in source
    assert "early_audit_state8_profile_ownership_checkpoint.py" in source
    assert "--permutation-replicates 2000" in source
    assert "strict_state_only'] is True" in source
    assert "python -u scripts/train_state8_profile_ownership_repair.py" not in source


def test_checkpoint_audit_config_pins_launcher() -> None:
    launcher = ROOT / "scripts/run_state8_profile_ownership_checkpoint_audit.sh"
    digest = hashlib.sha256(launcher.read_bytes()).hexdigest()
    config = (
        ROOT / "amlt/feeds_state8_profile_ownership_checkpoint200_audit_20260820.yaml"
    ).read_text()
    assert digest in config
    assert "80G1-A100" in config
    assert "rankfun_aml" in config
