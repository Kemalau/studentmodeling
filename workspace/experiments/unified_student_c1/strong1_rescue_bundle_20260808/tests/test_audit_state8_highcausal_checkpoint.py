from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_state8_highcausal_checkpoint import audit  # noqa: E402


def test_real_joint_round25_checkpoint_passes_if_available() -> None:
    checkpoint = Path("/tmp/joint25_recurrent_state.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 4_000_000_000:
        pytest.skip("live round-25 checkpoint is unavailable")
    value = audit(checkpoint, expected_round=25, variant="joint")
    assert value["status"] == "passed"
    assert value["history_rows"] == 100
    assert value["causal_exposures"] == 1000
    assert value["state_token_count"] == 8


def test_taxonomy_variant_is_audited_fail_closed() -> None:
    source = (ROOT / "scripts/audit_state8_highcausal_checkpoint.py").read_text()
    assert '"taxonomy_label_contract": "source_aware_no_answer_labels_v1"' in source
    assert '"taxonomy_candidates_visible_to_decoder": False' in source
    assert '"seed": 51' in source
    assert 'choices=("joint", "updater", "mid", "eedi", "taxonomy")' in source
