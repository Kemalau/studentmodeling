from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_teacher_gate import EXPECTED_SUPPORT, validate  # noqa: E402


def metrics() -> dict:
    groups = {}
    for name, n in EXPECTED_SUPPORT.items():
        groups[name] = {
            "n": n,
            "format_valid_rate": 1.0,
            "natural_schema_valid_rate": 1.0,
            "normalized_exact_match": 0.0,
            "llm_judge_accuracy": 0.6,
            "llm_judge_n": n,
            "llm_judge_api_or_cache": n,
            "llm_judge_exact_shortcuts": 0,
        }
    return {
        "generation_rows": sum(EXPECTED_SUPPORT.values()),
        "judged_rows": sum(EXPECTED_SUPPORT.values()),
        "judge_errors": 0,
        "judge_backend": "trapi",
        "judge_effective_model": "judge",
        "judge_rubric_sha256": "a" * 64,
        "metrics_by_group": groups,
    }


def write(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(value))
    return path


def test_teacher_gate_passes_semantic_judgments_without_exact_match(tmp_path: Path) -> None:
    result = validate(write(tmp_path, metrics()))
    assert result["pass"] is True
    assert result["exact_match_is_authoritative"] is False


def test_teacher_gate_rejects_exact_match_shortcut(tmp_path: Path) -> None:
    value = metrics()
    group = next(iter(value["metrics_by_group"].values()))
    group["llm_judge_api_or_cache"] -= 1
    group["llm_judge_exact_shortcuts"] = 1
    with pytest.raises(ValueError, match="non-semantic shortcut"):
        validate(write(tmp_path, value))


def test_teacher_gate_rejects_weak_eedi_teacher(tmp_path: Path) -> None:
    value = metrics()
    value["metrics_by_group"][
        "diagnose_misconception/eedi_mining/open_taxonomy_generation"
    ]["llm_judge_accuracy"] = 0.01
    assert validate(write(tmp_path, value))["pass"] is False


def test_teacher_gate_rejects_legacy_only_output(tmp_path: Path) -> None:
    value = metrics()
    group = value["metrics_by_group"][
        "diagnose_misconception/malrule/synthetic_same_template_open_generation"
    ]
    group["format_valid_rate"] = 1.0
    group["natural_schema_valid_rate"] = 0.0
    assert validate(write(tmp_path, value))["pass"] is False


def test_teacher_gate_rejects_non_trapi_backend(tmp_path: Path) -> None:
    value = metrics()
    value["judge_backend"] = "responses"
    with pytest.raises(ValueError, match="requires TRAPI"):
        validate(write(tmp_path, value))
