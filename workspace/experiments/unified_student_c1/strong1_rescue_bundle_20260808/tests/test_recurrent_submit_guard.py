from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from recurrent_submit_guard import (  # noqa: E402
    all_expected_jobs_pass,
    classify_experiment_status,
    has_active_job,
    has_forward_progress_job,
    paused_job_names,
    terminal_nonpass_job_names,
)


class RecurrentSubmitGuardTest(unittest.TestCase):
    def test_single_job_pass_table_is_terminal_success(self) -> None:
        status = """
JOB NAME                       DURATION  STATUS  SIZE
:m4-full-teacher-cache-seed42  2h        pass    13.6 kB
Experiment: recurrent-m4-cache
"""
        self.assertTrue(all_expected_jobs_pass(status, 1))

    def test_mixed_terminal_table_is_not_success(self) -> None:
        status = """
#   JOB NAME  DURATION  STATUS
:0  :first    2h        pass
:1  :second   2h        failed
"""
        self.assertFalse(all_expected_jobs_pass(status, 2))
        self.assertEqual(terminal_nonpass_job_names(status), ["second"])

    def test_terminal_nonpass_names_handle_numbered_and_plain_rows(self) -> None:
        status = """
#   JOB NAME                      DURATION  STATUS
:0  :cache-rank00                 2h        pass
:1  :cache-rank01                 2h        killed
:cache-rank02                     2h        failed
:3  :cache-rank03                 2h        cancelled
"""
        self.assertEqual(
            terminal_nonpass_job_names(status),
            ["cache-rank01", "cache-rank02", "cache-rank03"],
        )

    def test_aggregate_summary_remains_supported(self) -> None:
        self.assertTrue(all_expected_jobs_pass("4 jobs — 4 pass", 4))

    def test_paused_and_resuming_jobs_remain_active(self) -> None:
        for state in ("running", "queued", "preparing", "paused", "resuming"):
            self.assertTrue(has_active_job(f":job 2h {state} 2h ago"))
        self.assertFalse(has_active_job("2 jobs — 1 pass, 1 failed"))

    def test_paused_only_requires_explicit_recovery(self) -> None:
        status = """
#   JOB NAME       DURATION  STATUS
:0  :passed-shard  2h        pass
:1  :paused-shard  -         paused
"""
        self.assertTrue(has_active_job(status))
        self.assertFalse(has_forward_progress_job(status))
        self.assertEqual(paused_job_names(status), ["paused-shard"])
        self.assertTrue(has_forward_progress_job(":job 2h resuming 2h ago"))

    def test_existing_status_is_never_resubmitted(self) -> None:
        self.assertEqual(classify_experiment_status(0, "4 jobs — 4 pass"), "exists")

    def test_known_missing_messages_allow_one_submission(self) -> None:
        for message in (
            "Cannot find an experiment named x",
            "Experiment x not found",
            "No experiment exists with name x",
        ):
            self.assertEqual(classify_experiment_status(1, message), "missing")

    def test_auth_and_network_errors_fail_closed(self) -> None:
        for message in (
            "No authentication method succeeded",
            "management.azure.com timed out",
            "permission denied",
        ):
            self.assertEqual(classify_experiment_status(1, message), "ambiguous")


if __name__ == "__main__":
    unittest.main()
