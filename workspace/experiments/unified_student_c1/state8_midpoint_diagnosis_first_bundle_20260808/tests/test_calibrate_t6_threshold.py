from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_t6_threshold import fit, metrics  # noqa: E402


def row(probability: float, gold: str) -> dict:
    return {
        "task": "predict_error_persistence",
        "source": "s",
        "gold": gold,
        "probabilities": {"persistent": probability, "transient": 1 - probability},
    }


def test_fit_recovers_calibration_shift_without_test_input() -> None:
    rows = [
        row(0.46, "persistent"),
        row(0.44, "persistent"),
        row(0.42, "persistent"),
        row(0.40, "transient"),
        row(0.35, "transient"),
        row(0.30, "transient"),
    ]
    threshold, selected = fit(rows)
    assert 0.40 < threshold <= 0.42
    assert selected["macro_f1"] == 1.0
    assert metrics(rows, threshold)["positive_recall"] == 1.0
