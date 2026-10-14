import json
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_neural_response_specialist import train


def row(index: int, task: str, split: str, correct: int) -> dict:
    candidate_labels = ["a", "b"]
    target_label = candidate_labels[correct]
    target_response = "2" if correct else "1"
    target = f"<predicted_response>{target_response}</predicted_response>"
    if task == "predict_prospective_error":
        target = f"<correctness>{'correct' if correct else 'incorrect'}</correctness>" + target
    return {
        "example_id": f"{task}:{split}:{index}",
        "student_id": f"student-{index % 5}",
        "dataset": "fixture",
        "task": task,
        "history": [
            {
                "question_id": f"q-{index % 3}",
                "kc_ids": [f"k-{index % 2}"],
                "student_response_id": target_label,
                "correct_response_id": "b",
                "correctness": correct,
            }
        ],
        "query": {
            "question_id": f"q-{index % 3}",
            "kc_ids": [f"k-{index % 2}"],
            "correct_response_id": "b",
            "options": {"a": "1", "b": "2"},
        },
        "candidate_labels": candidate_labels,
        "candidate_targets": {
            "a": "<predicted_response>1</predicted_response>",
            "b": "<predicted_response>2</predicted_response>",
        },
        "target_label": target_label,
        "target": target,
        "metadata": {
            "mix_source": "fixture",
            "original_split": split,
            "eval_source": "fixture",
            "eval_protocol": "closed_choice_generation",
            "question_id": f"q-{index % 3}",
            "skill_id": f"k-{index % 2}",
            "target_correctness": correct,
        },
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def run_fixture(tmp_path: Path, task: str) -> dict:
    train_path = tmp_path / f"{task}.train.jsonl"
    test_path = tmp_path / f"{task}.test.jsonl"
    rows = [row(index, task, "valid" if index % 5 == 0 else "train", index % 2) for index in range(30)]
    write_rows(train_path, rows)
    write_rows(test_path, [row(index + 100, task, "test", index % 2) for index in range(8)])
    args = Namespace(
        task=task,
        train_jsonl=train_path,
        test_jsonl=test_path,
        output_dir=tmp_path / task,
        epochs=1,
        batch_size=8,
        eval_batch_size=8,
        embedding_dim=8,
        hidden_dim=16,
        dropout=0.0,
        learning_rate=1e-2,
        weight_decay=0.0,
        correctness_loss_weight=1.0,
        max_grad_norm=5.0,
        max_history=4,
        max_train_rows=None,
        max_test_rows=None,
        seed=42,
        device="cpu",
    )
    return train(args)


def test_t2_neural_response_specialist_writes_complete_receipt(tmp_path: Path) -> None:
    value = run_fixture(tmp_path, "predict_response")
    assert value["status"] == "complete"
    assert value["public_task"] == "T2"
    assert value["metrics"]["overall"]["n"] == 8
    assert len(value["artifacts"]["predictions_sha256"]) == 64


def test_t4_neural_response_specialist_uses_joint_math(tmp_path: Path) -> None:
    value = run_fixture(tmp_path, "predict_prospective_error")
    assert value["status"] == "complete"
    assert value["public_task"] == "T4"
    assert value["selection_metric"] == "joint_math_equivalence"
    assert value["metrics"]["overall"]["n"] == 8
    assert "correctness_accuracy" in value["metrics"]["overall"]
