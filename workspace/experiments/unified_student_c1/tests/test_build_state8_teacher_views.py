from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_state8_teacher_views import teacher_row  # noqa: E402


def recurrent_row(example_id: str = "x") -> dict:
    return {
        "example_id": example_id,
        "task": "diagnose_misconception",
        "source": "malrule",
        "target_label": "fractions.denominator_comparison_error",
        "target": "<misconception>fractions.denominator_comparison_error</misconception>",
        "direct_text": (
            "Return exactly one <misconception> tag.\n"
            "<student_history>wrong fraction comparison</student_history>\n"
            "<STUDENT_STATE>\n<query>diagnose</query>\nAnswer:"
        ),
        "update_texts": [],
        "original_metadata": {
            "eval_source": "malrule",
            "eval_protocol": "synthetic_same_template_open_generation",
        },
    }


def test_teacher_view_is_natural_full_history_and_keeps_label() -> None:
    row = teacher_row(recurrent_row())
    assert row is not None
    assert "<student_history>" in row["prompt"]
    assert "<STUDENT_STATE>" not in row["prompt"]
    assert "<label>fractions.denominator_comparison_error</label>" in row["target"]
    assert "larger denominator" in row["target"]
    assert "<diagnosis>" in row["target"] and "<evidence>" in row["target"]
    assert row["metadata"]["teacher_history_visible"] is True
    assert row["metadata"]["student_history_visible_at_final_inference"] is False


def test_raw_dataset_name_is_preserved_as_source() -> None:
    raw = recurrent_row()
    raw.pop("source")
    raw["dataset"] = "eedi_mining"
    row = teacher_row(raw)
    assert row is not None and row["source"] == "eedi_mining"


def test_builder_rejects_train_eval_overlap(tmp_path: Path) -> None:
    for name in ("train", "valid", "eval"):
        row = recurrent_row("overlap" if name != "valid" else "valid")
        (tmp_path / f"{name}.jsonl").write_text(json.dumps(row) + "\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_state8_teacher_views.py"),
            "--train-jsonl",
            str(tmp_path / "train.jsonl"),
            "--valid-jsonl",
            str(tmp_path / "valid.jsonl"),
            "--eval-jsonl",
            str(tmp_path / "eval.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "train/eval leakage" in result.stderr
