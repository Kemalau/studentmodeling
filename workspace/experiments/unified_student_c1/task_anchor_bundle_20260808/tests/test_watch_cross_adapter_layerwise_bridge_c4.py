from __future__ import annotations

import unittest
from pathlib import Path


class CrossAdapterBridgeWatcherTest(unittest.TestCase):
    def test_watcher_is_fail_closed_and_no_resubmission(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/watch_cross_adapter_layerwise_bridge_c4.sh").read_text()
        self.assertIn("all_expected_jobs_pass(sys.argv[1], 4)", text)
        self.assertIn("capture_adapter", text)
        self.assertIn("decode_adapter", text)
        self.assertIn("raw_history_visible_to_decoder", text)
        self.assertIn("touch \"$state_dir/needs_attention\"", text)
        self.assertNotIn("amlt run", text)


if __name__ == "__main__":
    unittest.main()
