import json
from pathlib import Path

import pytest

from scripts import materialize_taskonly_qwen_splits as module


def row(task: str, example_id: str, original_id: str, split: str) -> dict:
    return {
        "example_id": example_id,
        "task": task,
        "dataset": "source",
        "metadata": {
            "mix_source": "source",
            "eval_source": "source",
            "original_example_id": original_id,
            "original_split": split,
        },
    }


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


def test_materialization_is_task_pure_and_rejects_test_overlap(tmp_path: Path, monkeypatch) -> None:
    tasks = list(module.EXPECTED_TRAIN)
    monkeypatch.setattr(module, "EXPECTED_TRAIN", {task: 1 for task in tasks})
    monkeypatch.setattr(module, "EXPECTED_TEST", {task: 1 for task in tasks})
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    write(train, [row(task, f"train-{task}", f"train-original-{task}", "train") for task in tasks])
    write(test, [row(task, f"test-{task}", f"test-original-{task}", "test") for task in tasks])
    receipt = module.materialize(train, test, tmp_path / "out")
    assert receipt["test_used_for_training"] is False
    for task in tasks:
        values = [json.loads(line) for line in (tmp_path / "out" / f"{task}.jsonl").read_text().splitlines()]
        assert [value["task"] for value in values] == [task]
        assert receipt["tasks"][task]["train_test_original_id_overlap"] == 0

    write(test, [row(task, f"test-{task}", f"train-original-{task}" if task == tasks[0] else f"test-original-{task}", "test") for task in tasks])
    with pytest.raises(ValueError, match="leakage"):
        module.materialize(train, test, tmp_path / "out-overlap")
