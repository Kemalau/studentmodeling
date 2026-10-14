from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchPromotedRound1250C4MultiseedTest(unittest.TestCase):
    def test_monitor_is_fail_closed_and_aggregates_three_seeds(self) -> None:
        source = (ROOT / "scripts/watch_promoted_round1250_c4_n150_multiseed.sh").read_text()
        self.assertIn("promoted-round1250-c4-n150-multiseed-20260804", source)
        self.assertIn("all_expected_jobs_pass(sys.argv[1], int(sys.argv[2]))", source)
        self.assertIn("{13,42,97}", source)
        self.assertIn("row['checkpoint_round']==1250", source)
        self.assertIn("amlt logs view -n 1000", source)
        self.assertIn('touch "$state_dir/needs_attention"', source)
        self.assertIn("two_of_three_gate_pass", source)
        self.assertIn("terminal_nonpass_job_names", source)
        self.assertIn("preserve_passed", source)
        self.assertIn("failed-sibling recovery", source)
        self.assertIn("feeds_submission_global.lock", source)


if __name__ == "__main__":
    unittest.main()
