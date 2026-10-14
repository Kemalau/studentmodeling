from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RehearsalWatcherTest(unittest.TestCase):
    def test_watcher_has_recovery_and_scientific_summary(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_rehearsal.sh").read_text(encoding="utf-8")
        for required in (
            "feeds_submission_global.lock",
            "validate_m4_causal_rehearsal_amlt_config.py",
            "terminal_nonpass_job_names",
            "--expected-branches repeat100_w1 repeat100_w4",
            "summarize_m4_causal_strength_branches.py",
            "touch \"$state_dir/complete\"",
            "timeout 180s amlt logs view",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
