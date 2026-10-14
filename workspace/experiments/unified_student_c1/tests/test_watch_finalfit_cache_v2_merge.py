from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FinalFitCacheV2MergeWatcherTest(unittest.TestCase):
    def test_watcher_is_fail_closed_and_publishes_atomically(self) -> None:
        text = (ROOT / "scripts/watch_finalfit_cache_v2_merge.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "all_expected_jobs_pass",
            "timeout 180s amlt logs view",
            "amlt results download",
            "partition_count'] == 16",
            "integrity['status'] == 'verified'",
            'rm -f "$state_dir/needs_attention"',
            'touch "$state_dir/complete"',
        ):
            self.assertIn(required, text)
        self.assertLess(
            text.index('rm -f "$state_dir/needs_attention"'),
            text.index("while true; do"),
        )


if __name__ == "__main__":
    unittest.main()
