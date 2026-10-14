from __future__ import annotations

from pathlib import Path

from scripts.validate_state8_exact_diagnosis_pipeline_amlt import validate


ROOT = Path(__file__).resolve().parents[1]


def test_exact_pipeline_amlt_passes() -> None:
    validate(ROOT / "amlt/feeds_state8_exact_diagnosis_pipeline_20260808.yaml")
