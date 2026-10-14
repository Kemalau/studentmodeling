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
    assert result["terminal_experiments_recovered_by_full_status"] == []


def test_status_queries_force_refresh_instead_of_trusting_stale_local_cache() -> None:
    responses = [
        [{"experiment_name": "stale-running"}],
        ValueError("filtered status became empty after refresh"),
        [{"job_name": ":old", "status": "killed", "sku": "80G1-A100"}],
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses) as mocked:
        result = count(32, 400)
    assert result["running_a100_requests"] == 0
    assert result["terminal_experiments_recovered_by_full_status"] == ["stale-running"]
    filtered, fallback = mocked.call_args_list[1].args[0], mocked.call_args_list[2].args[0]
    assert "--force-status-update" in filtered
    assert "--force-status-update" in fallback


def test_queued_only_experiment_is_discovered_and_counted() -> None:
    responses = [
        [{
            "experiment_name": "queued-only",
            "_statuses": ["queued"],
            "_all_terminated": False,
        }],
        [{"job_name": ":q", "job_id": "q", "sku": "80G2-A100"}],
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["running_a100_requests"] == 2
    assert result["available_user_slots"] == 30


def test_terminal_summaries_are_skipped_before_status_calls() -> None:
    responses = [[
        {
            "experiment_name": "done",
            "_statuses": ["pass"],
            "_all_terminated": True,
        },
        {
            "experiment_name": "mixed",
            "_statuses": ["pass", "queued"],
            "_all_terminated": False,
        },
    ], [
        {"job_name": ":done", "job_id": "done", "sku": "80G1-A100"},
    ]]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses) as mocked:
        result = count(32, 400)
    assert mocked.call_count == 2
    assert result["running_a100_requests"] == 1
    assert result["jobs"][0]["experiment"] == "mixed"


def test_unreadable_running_experiment_closes_slot_estimate() -> None:
    responses = [
        [{"experiment_name": "known"}, {"experiment_name": "stale"}],
        [{"job_name": ":a", "job_id": "a", "sku": "80G1-A100"}],
        ValueError("missing JSON table"),
        ValueError("fallback missing JSON table"),
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["running_a100_requests"] == 1
    assert result["available_user_slots"] == 0
    assert result["auditable"] is False
    assert result["unreadable_experiments"][0]["experiment"] == "stale"


def test_stale_running_list_entry_is_recovered_only_when_full_status_is_terminal() -> None:
    responses = [
        [{"experiment_name": "known"}, {"experiment_name": "stale"}],
        [{"job_name": ":a", "job_id": "a", "sku": "80G1-A100"}],
        ValueError("missing filtered JSON table"),
        [{"job_name": ":old", "status": "failed", "sku": "80G4-A100"}],
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["running_a100_requests"] == 1
    assert result["available_user_slots"] == 31
    assert result["auditable"] is True
    assert result["unreadable_experiments"] == []
    assert result["terminal_experiments_recovered_by_full_status"] == ["stale"]


def test_filtered_failure_uses_active_rows_from_readable_full_status() -> None:
    responses = [
        [{"experiment_name": "stale"}],
        ValueError("missing filtered JSON table"),
        [{"job_name": ":queued", "status": "queued", "sku": "80G4-A100"}],
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["running_a100_requests"] == 4
    assert result["available_user_slots"] == 28
    assert result["auditable"] is True
    assert result["terminal_experiments_recovered_by_full_status"] == []


def test_unknown_full_status_remains_fail_closed() -> None:
    responses = [
        [{"experiment_name": "stale"}],
        ValueError("missing filtered JSON table"),
        [{"job_name": ":odd", "status": "mystery", "sku": "80G4-A100"}],
    ]
    with patch("scripts.count_active_a100_requests.json_table", side_effect=responses):
        result = count(32, 400)
    assert result["available_user_slots"] == 0
    assert result["auditable"] is False
