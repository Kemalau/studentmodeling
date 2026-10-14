from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_state8_diagnosis_first_teacher_views import teacher_row  # noqa: E402


def recurrent_row() -> dict:
    return {
        "example_id": "diagnosis-first-teacher-test",
        "task": "diagnose_misconception",
        "source": "malrule",
        "target_label": "fractions.denominator_comparison_error",
        "target": (
            "<misconception>fractions.denominator_comparison_error</misconception>"
        ),
        "direct_text": (
            "Return exactly one <misconception> tag.\n"
            "<student_history>the student orders fractions only by denominator"
            "</student_history>\n<STUDENT_STATE>\n"
            "<query>Diagnose the misconception.</query>\nAnswer:"
        ),
        "update_texts": [],
        "original_metadata": {
            "eval_source": "malrule",
            "eval_protocol": "synthetic_same_template_open_generation",
        },
    }


def test_teacher_and_student_share_diagnosis_first_schema() -> None:
    row = teacher_row(recurrent_row())
    assert row is not None
    target = row["target"]
    assert target.index("<diagnosis>") < target.index("<evidence>")
    assert target.index("<evidence>") < target.index("<label>")
    assert "<label>fractions.denominator_comparison_error</label>" in target
    assert "larger denominator" in target
    assert "<STUDENT_STATE>" not in row["prompt"]
    assert row["metadata"]["diagnosis_field_order"] == [
        "diagnosis",
        "evidence",
        "label",
    ]
    assert row["metadata"]["semantic_fields_primary"] is True
