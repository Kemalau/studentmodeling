from __future__ import annotations

import pytest

from scripts.train_state8_task_anchored_causal_repair import (
    anchor_phase,
    is_diagnosis,
    parse_schedule,
)


def test_anchor_schedule_is_monotone_and_three_stage() -> None:
    assert parse_schedule("1.0,0.5,0.25") == (1.0, 0.5, 0.25)
    assert anchor_phase(0.0) == 0
    assert anchor_phase(0.3999) == 0
    assert anchor_phase(0.4) == 1
    assert anchor_phase(0.6999) == 1
    assert anchor_phase(0.7) == 2
    assert anchor_phase(1.0) == 2


def test_anchor_rejects_invalid_schedules() -> None:
    with pytest.raises(ValueError):
        parse_schedule("0.5,1.0,0.25")
    with pytest.raises(ValueError):
        parse_schedule("1.0,0.0,0.0")


def test_only_diagnosis_rows_receive_the_strong_anchor() -> None:
    assert is_diagnosis({"task": "diagnose_misconception"}) is True
    assert is_diagnosis({"task": "predict_error_persistence"}) is False
