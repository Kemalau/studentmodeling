from __future__ import annotations

import unittest
from pathlib import Path


class WatchLayerwiseFormalScreenTest(unittest.TestCase):
    def test_watcher_is_fail_closed_and_audits_information_route(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/watch_layerwise_formal_screen.sh").read_text()
        self.assertIn("all_expected_jobs_pass(sys.argv[1], 4)", text)
        self.assertIn("raw_history_visible_to_decoder", text)
        self.assertIn("scientific_gate_pass", text)
        self.assertIn("format_valid_rate", text)
        self.assertIn("persistent_history_kv_cache", text)
        self.assertIn("optimizer_steps=0", text)
        self.assertIn("amlt logs view", text)
        self.assertIn("needs_attention", text)
        self.assertIn("recovery1", text)


if __name__ == "__main__":
    unittest.main()
