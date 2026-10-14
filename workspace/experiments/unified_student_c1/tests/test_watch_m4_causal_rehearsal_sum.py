from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SumAlignedRehearsalWatcherTest(unittest.TestCase):
    def test_watcher_is_recoverable_and_fail_closed(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_rehearsal_sum.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "feeds_submission_global.lock",
            "validate_m4_causal_rehearsal_amlt_config.py",
            "classify_experiment_status",
            "terminal_nonpass_job_names",
            "--expected-branches repeat100_sum_w1 repeat100_sum_w4",
            "summarize_m4_causal_strength_branches.py",
            'touch "$state_dir/complete"',
            "timeout 180s amlt logs view",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
