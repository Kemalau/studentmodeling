from __future__ import annotations
import unittest
from pathlib import Path


class LayerwiseRecurrentC4WatcherTest(unittest.TestCase):
    def test_watcher_is_fail_closed(self) -> None:
        root=Path(__file__).resolve().parents[1]
        text=(root/'scripts/watch_layerwise_recurrent_c4.sh').read_text()
        self.assertIn('all_expected_jobs_pass(sys.argv[1], 3)',text)
        self.assertIn('final_layerwise_recurrent_state',text)
        self.assertIn('raw_history_visible_to_decoder',text)
        self.assertIn('touch "$state_dir/needs_attention"',text)
        self.assertNotIn('amlt run',text)


if __name__=='__main__': unittest.main()
