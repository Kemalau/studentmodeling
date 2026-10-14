from __future__ import annotations

from unittest.mock import patch

from scripts.count_active_a100_requests import count


def test_counts_only_running_a100_slots() -> None:
    responses = [
        [{"experiment_name": "one"}, {"experiment_name": "two"}],
        [
            {"job_name": ":a", "job_id": "a", "sku": "80G2-A100"},
            {"job_name": ":cpu", "job_id": "cpu", "sku": "Dv3"},
        ],
        [{"job_name": ":b", "job_id": "b", "sku": "80G1-A100"}],
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["running_a100_requests"] == 3
    assert result["available_user_slots"] == 29
    assert len(result["jobs"]) == 2
    assert result["auditable"] is True
    assert result["unreadable_experiments"] == []


def test_unreadable_running_experiment_closes_slot_estimate() -> None:
    responses = [
        [{"experiment_name": "known"}, {"experiment_name": "stale"}],
        [{"job_name": ":a", "job_id": "a", "sku": "80G1-A100"}],
        ValueError("missing JSON table"),
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["running_a100_requests"] == 1
    assert result["available_user_slots"] == 0
    assert result["auditable"] is False
    assert result["unreadable_experiments"][0]["experiment"] == "stale"
