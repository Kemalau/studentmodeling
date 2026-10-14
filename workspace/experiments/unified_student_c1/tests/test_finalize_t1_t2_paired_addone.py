from __future__ import annotations

import pytest

from scripts.finalize_t1_t2_paired_addone import (
    binary_auroc,
    challenger_view,
    wilson_interval,
)


def test_binary_auroc_handles_ties() -> None:
    assert binary_auroc([0, 1, 0, 1], [0.1, 0.9, 0.5, 0.5]) == pytest.approx(0.875)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(70, 100)
    assert low < 0.7 < high
    assert low == pytest.approx(0.604151, abs=1e-6)
    assert high == pytest.approx(0.781052, abs=1e-6)


def test_challenger_view_renames_direction_without_changing_values() -> None:
    raw = {
        "n": 4,
        "taskonly_primary": 0.75,
        "unified_primary": 0.5,
        "taskonly_minus_unified": 0.25,
        "paired_bootstrap_95ci": [0.0, 0.5],
        "discordant": {
            "taskonly_only_correct": 2,
            "unified_only_correct": 1,
            "both_correct": 1,
            "both_wrong": 0,
        },
        "mcnemar_exact_two_sided_p": 1.0,
        "example_id_set_sha256": "a" * 64,
    }
    value = challenger_view(raw, "T1-only")
    assert value["challenger_primary"] == 0.75
    assert value["reference_primary"] == 0.5
    assert value["challenger_minus_reference"] == 0.25
    assert value["discordant"]["challenger_only_correct"] == 2
