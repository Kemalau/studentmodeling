from __future__ import annotations

import unittest
from pathlib import Path


class LayerwiseStateBridgeWatcherTest(unittest.TestCase):
    def test_watcher_is_fail_closed_and_checks_history_invariants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/watch_layerwise_state_bridge_c4.sh").read_text()
        self.assertIn("all_expected_jobs_pass(sys.argv[1], 2)", text)
        self.assertIn("raw_history_visible_to_decoder", text)
        self.assertIn("persistent_history_kv_cache", text)
        self.assertIn("optimizer_steps", text)
        self.assertIn("touch \"$state_dir/needs_attention\"", text)
        self.assertNotIn("amlt run", text)


if __name__ == "__main__":
    unittest.main()
