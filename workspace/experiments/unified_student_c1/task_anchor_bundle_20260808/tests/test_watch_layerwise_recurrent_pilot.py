from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchLayerwiseRecurrentPilotTest(unittest.TestCase):
    def test_controller_is_fail_closed_and_gate_ordered(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_recurrent_pilot.sh").read_text()
        for experiment in (
            "recurrent-layerwise-pilot-preserve-low-lr-seed42-20260731-split3",
            "recurrent-layerwise-pilot-balanced-seed42-20260731-split3",
            "recurrent-layerwise-pilot-specificity-seed42-20260731-split3",
        ):
            self.assertIn(experiment, text)
        self.assertIn('for experiment in "${experiments[@]}"', text)
        self.assertIn("all_expected_jobs_pass(sys.argv[1], 1)", text)
        self.assertIn('passed == ${#experiments[@]}', text)
        self.assertIn("select_layerwise_recurrent_pilot.py", text)
        self.assertIn("scientific_gate_failed", text)
        self.assertIn('fail "split layerwise pilot did not pass', text)


if __name__ == "__main__":
    unittest.main()
