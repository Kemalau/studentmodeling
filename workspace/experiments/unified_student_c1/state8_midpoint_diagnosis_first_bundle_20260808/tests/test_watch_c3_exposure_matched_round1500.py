from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchC3ExposureMatchedTest(unittest.TestCase):
    def test_controller_recovers_siblings_and_separates_completion_from_effect(self) -> None:
        source = (ROOT / "scripts/watch_c3_exposure_matched_round1500.sh").read_text()
        self.assertIn("c3-exposure-matched-round1500-seed20260804", source)
        self.assertIn("c3_exposure_matched_round1500_hashfix_v2", source)
        self.assertIn("c3-hashfix-v2-round1500", source)
        self.assertIn("has_active_job", source)
        self.assertIn("all_expected_jobs_pass", source)
        self.assertIn("validate_c3_exposure_matched_amlt.py", source)
        self.assertIn("feeds_submission_global.lock", source)
        self.assertIn("failed-sibling recovery", source)
        self.assertIn("select_azure_profile", source)
        self.assertIn("auth_required", source)
        self.assertIn("/home/v-yuejiali/.azure", source)
        self.assertIn("/home/v-yuejiali/.azure-studentmodeling-feeds", source)
        self.assertIn("judge_formal_t3_predictions.py", source)
        self.assertIn("summarize_c3_exposure_matched.py", source)
        self.assertIn("scientific_gate_failed", source)
        self.assertIn('touch "$state_dir/complete"', source)


if __name__ == "__main__":
    unittest.main()
