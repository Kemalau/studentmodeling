from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_diagnosis_first_teacher_screen import EXPECTED, validate  # noqa: E402


def test_semantic_screen_is_authoritative_and_exact_match_is_not(tmp_path: Path) -> None:
    groups = {}
    for name, (support, threshold) in EXPECTED.items():
        groups[name] = {
            "n": support,
            "llm_judge_n": support,
            "llm_judge_api_or_cache": support,
            "llm_judge_accuracy": threshold,
            "format_valid_rate": 1.0,
            "natural_schema_valid_rate": 1.0,
        }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "judge_backend": "trapi",
        "judge_errors": 0,
        "generation_rows": 75,
        "judged_rows": 75,
        "metrics_by_group": groups,
    }))
    result = validate(path)
    assert result["pass"] is True
    assert result["semantic_judge_authoritative"] is True
    assert result["exact_match_authoritative"] is False
    assert result["diagnosis_field_order"] == ["diagnosis", "evidence", "label"]
