from __future__ import annotations

from scripts.train_state8_interleaved_retention_repair import (
    audit_optimizer_order,
    identity_real_schedule,
)


def row(index: int, task: str, target: str = "") -> dict:
    return {
        "example_id": f"row:{index}",
        "task": task,
        "source": "test",
        "target": target,
    }


def test_identity_real_schedule_retains_causal_schedule() -> None:
    def original(real_count: int, causal_count: int, seed: int):
        assert (real_count, causal_count, seed) == (4, 2, 7)
        return [2, 0, 3, 1], {0: 1, 3: 0}

    real, causal = identity_real_schedule(original, 4, 2, 7)
    assert real == [0, 1, 2, 3]
    assert causal == {0: 1, 3: 0}


def test_optimizer_order_audit_checks_balanced_t6_prefix() -> None:
    rows = [
        row(0, "diagnose_misconception"),
        row(1, "predict_error_persistence", "<persistence>persistent</persistence>"),
        row(2, "predict_error_persistence", "<persistence>transient</persistence>"),
    ]
    rows.extend(row(i, "predict_correctness") for i in range(3, 20))
    history = [
        {
            "example_id": item["example_id"],
            "task": item["task"],
            "global_position": index,
        }
        for index, item in enumerate(rows)
    ]
    receipt = audit_optimizer_order(rows, history)
    assert receipt["optimizer_order_exact"] is True
    assert receipt["prefix_audit"]["20"]["t6_label_counts"] == {
        "persistent": 1,
        "transient": 1,
    }


def test_optimizer_order_audit_rejects_shuffle() -> None:
    rows = [row(0, "predict_correctness"), row(1, "predict_response")]
    history = [
        {"example_id": "row:1", "task": "predict_response", "global_position": 0},
        {"example_id": "row:0", "task": "predict_correctness", "global_position": 1},
    ]
    try:
        audit_optimizer_order(rows, history)
    except ValueError as error:
        assert "Optimizer order differs" in str(error)
    else:
        raise AssertionError("Expected shuffled optimizer order to fail")
