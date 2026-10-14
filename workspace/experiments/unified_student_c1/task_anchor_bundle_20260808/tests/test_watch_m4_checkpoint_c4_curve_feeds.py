from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CheckpointC4CurveWatcherTest(unittest.TestCase):
    def test_probes_latest_durable_checkpoint_and_tolerates_visibility_lag(self) -> None:
        text = (ROOT / "scripts/watch_m4_checkpoint_c4_curve_feeds.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("rounds=(1000 1750 2000 3000)", text)
        self.assertIn("No jobs found in the current experiment", text)
        self.assertIn("sleep 30", text)


if __name__ == "__main__":
    unittest.main()
