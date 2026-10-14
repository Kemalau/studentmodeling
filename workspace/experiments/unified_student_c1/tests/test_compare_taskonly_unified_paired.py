from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_taskonly_unified_paired as paired  # noqa: E402


def test_binary_paired_comparison_counts_discordance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(paired.TASKS, "T1", ("predict_correctness", 4))
    unified = [
        {"example_id": str(i), "task": "predict_correctness", "gold": "correct", "prediction": pred}
        for i, pred in enumerate(["correct", "correct", "incorrect", "incorrect"])
    ]
    taskonly = [
        {"example_id": str(i), "task": "predict_correctness", "gold": "correct", "prediction": pred}
        for i, pred in enumerate(["correct", "incorrect", "correct", "correct"])
    ]
    value = paired.compare(unified, taskonly, "T1", bootstrap_samples=200, seed=1)
    assert value["taskonly_primary"] == 0.75
    assert value["unified_primary"] == 0.5
    assert value["discordant"]["taskonly_only_correct"] == 2
    assert value["discordant"]["unified_only_correct"] == 1


def test_t3_fails_closed_without_semantic_judgment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(paired.TASKS, "T3", ("diagnose_misconception", 1))
    rows = [{"example_id": "x", "task": "diagnose_misconception"}]
    with pytest.raises(ValueError, match="semantic judgment"):
        paired.compare(rows, rows, "T3", bootstrap_samples=10)


def test_paired_comparison_rejects_different_example_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(paired.TASKS, "T5", ("predict_error_persistence", 1))
    unified = [{"example_id": "u", "task": "predict_error_persistence", "gold": "persistent", "prediction": "persistent"}]
    taskonly = [{"example_id": "s", "task": "predict_error_persistence", "gold": "persistent", "prediction": "persistent"}]
    with pytest.raises(ValueError, match="example-ID sets differ"):
        paired.compare(unified, taskonly, "T5", bootstrap_samples=10)
