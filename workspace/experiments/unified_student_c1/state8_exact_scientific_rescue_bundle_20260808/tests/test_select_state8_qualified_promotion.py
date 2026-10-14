from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_state8_qualified_promotion import select  # noqa: E402


def gate(path: Path, variant: str, passed: bool) -> Path:
    stage = {
        "t3_heavy": "state8_qualified_promotion_gate_v1",
        "highkd": "state8_qualified_highkd_promotion_gate_v1",
        "balanced": "state8_qualified_balanced_promotion_gate_v1",
    }[variant]
    checks = {"C4_multiseed": passed, "T3_semantic": passed, "T1_T2_T5_T6_small_screen": passed}
    path.write_text(json.dumps({"stage": stage, "promotion_variant": variant, "pass": passed, "checks": checks, "artifact_sha256": {"adapter_model": "a" * 64, "recurrent_state": "b" * 64}, "C4": {"rows": [{"pair_seed": x} for x in (13, 42, 97)]}}))
    return path


def test_primary_is_preferred_when_both_pass(tmp_path: Path) -> None:
    value = select(gate(tmp_path / "p.json", "t3_heavy", True), gate(tmp_path / "h.json", "highkd", True), gate(tmp_path / "b.json", "balanced", True))
    assert value["selected"]["variant"] == "t3_heavy"


def test_primary_pass_short_circuits_unfinished_hedges(tmp_path: Path) -> None:
    value = select(gate(tmp_path / "p.json", "t3_heavy", True))
    assert value["selected"]["variant"] == "t3_heavy"
    assert value["arms"]["highkd"]["status"] == "not_required_by_preregistered_priority"
    assert value["arms"]["balanced"]["status"] == "not_required_by_preregistered_priority"


def test_highkd_is_used_when_primary_fails(tmp_path: Path) -> None:
    value = select(gate(tmp_path / "p.json", "t3_heavy", False), gate(tmp_path / "h.json", "highkd", True))
    assert value["selected"]["variant"] == "highkd"


def test_failed_primary_requires_highkd_gate(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="high-KD gate is required"):
        select(gate(tmp_path / "p.json", "t3_heavy", False))


def test_balanced_is_used_only_when_primary_and_highkd_fail(tmp_path: Path) -> None:
    value = select(gate(tmp_path / "p.json", "t3_heavy", False), gate(tmp_path / "h.json", "highkd", False), gate(tmp_path / "b.json", "balanced", True))
    assert value["selected"]["variant"] == "balanced"


def test_no_passing_arm_withholds_selection(tmp_path: Path) -> None:
    value = select(gate(tmp_path / "p.json", "t3_heavy", False), gate(tmp_path / "h.json", "highkd", False), gate(tmp_path / "b.json", "balanced", False))
    assert value["pass"] is False and value["selected"] is None
