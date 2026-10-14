from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_selector_requires_strict_unified_and_dual_c4_gate() -> None:
    source = (ROOT / "scripts/select_state8_midpoint_diagnosis_first.py").read_text()
    for fragment in (
        '"T3_eedi_semantic": 0.17',
        '"T3_synthetic_semantic": 0.45',
        '"diagnosis_C4_swap_to_donor": 0.50',
        '"diagnosis_C4_random_stable_max": 0.25',
        'checks["answer_C4_pass"] and checks["diagnosis_C4_pass"]',
        'initialization_only_not_complete_curriculum',
    ):
        assert fragment in source


def test_watcher_recovers_remote_failures_and_uses_trapi() -> None:
    source = (ROOT / "scripts/watch_state8_midpoint_diagnosis_first.sh").read_text()
    assert "submit_recovery" in source
    assert "amlt resume" in source
    assert "--judge-backend trapi" in source
    assert "summarize_state8_diagnosis_c4.py" in source
