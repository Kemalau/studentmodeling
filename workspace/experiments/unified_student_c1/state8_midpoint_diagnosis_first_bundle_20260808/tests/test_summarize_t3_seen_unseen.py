import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_t3_seen_unseen import build  # noqa: E402


def row(example_id: str, label: str, *, output: str = "<misconception>x</misconception>", correct: bool = False):
    return {
        "example_id": example_id,
        "task": "diagnose_misconception",
        "source": "eedi_mining",
        "target_label": label,
        "raw_output": output,
        "llm_judge_correct": correct,
    }


def test_seen_unseen_split_and_output_collapse() -> None:
    train = [row("train-a", "a"), row("train-b", "b")]
    eval_rows = [row("seen", "a"), row("unseen-1", "c"), row("unseen-2", "d")]
    predictions = [
        row("seen", "a", output="<misconception>good</misconception>"),
        row("unseen-1", "c", output="collapsed"),
        row("unseen-2", "d", output="collapsed"),
    ]
    judgments = [row("seen", "a", correct=True), row("unseen-1", "c"), row("unseen-2", "d")]
    result = build(train, eval_rows, predictions, judgments)
    assert result["evaluation_seen_label_rows"] == 1
    assert result["evaluation_unseen_label_rows"] == 2
    assert result["seen_label"]["semantic_accuracy"] == 1.0
    assert result["unseen_label"]["semantic_accuracy"] == 0.0
    assert result["unseen_label"]["unique_outputs"] == 1
    assert result["unseen_label"]["top_output_share"] == 1.0


def test_missing_judgment_fails_closed() -> None:
    train = [row("train-a", "a")]
    eval_rows = [row("test", "a")]
    predictions = [row("test", "a")]
    try:
        build(train, eval_rows, predictions, [])
    except ValueError as error:
        assert "lacks semantic judgment" in str(error)
    else:
        raise AssertionError("missing semantic judgment was accepted")
