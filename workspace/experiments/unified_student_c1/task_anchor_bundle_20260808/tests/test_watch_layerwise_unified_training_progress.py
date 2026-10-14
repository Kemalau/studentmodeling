from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TrainingProgressWatcherTest(unittest.TestCase):
    def test_health_alert_is_advisory_and_stall_remains_separate(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_unified_training_progress.sh").read_text()
        self.assertIn("health_alert.json", text)
        self.assertIn("stalled.json", text)
        self.assertIn("A stale pre-authentication stall receipt", text)
        self.assertIn('rm -f "$state_dir/stalled.json"', text)
        self.assertIn('timeout 90 amlt status', text)
        self.assertIn('timeout 90 amlt logs list', text)
        self.assertIn("--awaiting-first-record", text)
        self.assertIn('progress_experiment" != "$experiment', text)
        self.assertIn('progress_round_state" == none', text)
        self.assertIn('FIT_DIR:-$pipeline/final_fit', text)
        self.assertIn('STATE_DIR:-$pipeline/progress', text)
        self.assertIn('TOTAL_ROUNDS:-16241', text)
        self.assertNotIn("amlt cancel", text)
        self.assertNotIn("amlt run", text)


if __name__ == "__main__":
    unittest.main()
