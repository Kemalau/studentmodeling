import json

import pytest

from scripts.train_c1_lora import read_jsonl


def test_read_jsonl_excludes_one_task_before_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"task": "a", "value": 1},
        {"task": "b", "value": 2},
        {"task": "a", "value": 3},
        {"task": "c", "value": 4},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert read_jsonl(path, limit=2, exclude_task="a") == [rows[1], rows[3]]


def test_read_jsonl_rejects_include_and_exclude_together(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"task": "a"}) + "\n")

    with pytest.raises(ValueError, match="either task_filter or exclude_task"):
        read_jsonl(path, task_filter="a", exclude_task="b")
