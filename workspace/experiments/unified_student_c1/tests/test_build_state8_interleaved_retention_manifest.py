from __future__ import annotations

from collections import Counter

from scripts.build_state8_interleaved_retention_manifest import build


def row(index: int, task: str, source: str, target: str = "") -> dict:
    return {
        "example_id": f"train:{index}",
        "task": task,
        "source": source,
        "target": target,
    }


def test_interleave_keeps_support_and_balances_early_prefixes() -> None:
    rows = []
    rows.extend(row(i, "diagnose_misconception", "eedi") for i in range(50))
    rows.extend(
        row(i, "predict_error_persistence", "fa", "<persistence>persistent</persistence>")
        for i in range(50, 65)
    )
    rows.extend(
        row(i, "predict_error_persistence", "fa", "<persistence>transient</persistence>")
        for i in range(65, 80)
    )
    rows.extend(row(i, "predict_correctness", "fa") for i in range(80, 100))
    source = {
        "stage": "layerwise_task_retention_index_manifest_v1",
        "branch": "staged_t3_t6",
        "segment_rows": 100,
        "indices": list(range(100)),
        "selected_example_ids_sha256": "source-order",
    }
    value = build(source, rows, [{"example_id": "formal:0"}])
    assert set(value["indices"]) == set(range(100))
    assert value["support_unchanged"] is True
    first_twenty = [rows[index] for index in value["indices"][:20]]
    tasks = Counter(item["task"] for item in first_twenty)
    assert tasks == {
        "diagnose_misconception": 10,
        "predict_error_persistence": 6,
        "predict_correctness": 4,
    }
    labels = Counter(item["target"] for item in first_twenty if item["target"])
    assert labels["<persistence>persistent</persistence>"] == 3
    assert labels["<persistence>transient</persistence>"] == 3


def test_interleave_rejects_formal_overlap() -> None:
    rows = [row(i, "predict_correctness", "fa") for i in range(2)]
    source = {
        "stage": "layerwise_task_retention_index_manifest_v1",
        "branch": "staged_t3_t6",
        "segment_rows": 2,
        "indices": [0, 1],
        "selected_example_ids_sha256": "x",
    }
    try:
        build(source, rows, [{"example_id": "train:1"}])
    except ValueError as error:
        assert "train/test overlap" in str(error)
    else:
        raise AssertionError("Expected overlap rejection")
