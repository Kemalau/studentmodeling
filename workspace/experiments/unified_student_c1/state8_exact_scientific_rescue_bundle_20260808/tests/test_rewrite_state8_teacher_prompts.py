from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rewrite_state8_teacher_prompts import rewrite  # noqa: E402


def row(index: int, source: str) -> dict:
    return {
        "example_id": f"e{index}",
        "task": "diagnose_misconception",
        "source": source,
        "prompt": (
            "Return exactly <label>, <diagnosis>, and <evidence> tags.\n"
            "Output format: <misconception>systematic misconception</misconception>\n"
            "Answer:"
        ),
        "direct_text": "same",
        "target": "<label>x</label>\n<diagnosis>y</diagnosis>\n<evidence>z</evidence>",
    }


def test_rewrite_frozen_panel_removes_legacy_contract(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    rows = [row(i, "eedi_mining" if i < 894 else "malrule") for i in range(1494)]
    source.write_text("".join(json.dumps(value) + "\n" for value in rows))
    receipt = rewrite(source, output)
    assert receipt["rows"] == 1494
    assert receipt["changed_rows"] == 1494
    assert receipt["legacy_answer_adjacent_contracts"] == 0
    text = output.read_text()
    assert "Output format: <misconception>" not in text
    assert "<diagnosis>natural-language diagnosis</diagnosis>" in text


def test_rewrite_rejects_incomplete_target_schema(tmp_path: Path) -> None:
    source = tmp_path / "input.jsonl"
    value = row(0, "eedi_mining")
    value["target"] = "<diagnosis>only</diagnosis>"
    source.write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="Target schema drift"):
        rewrite(source, tmp_path / "output.jsonl")
