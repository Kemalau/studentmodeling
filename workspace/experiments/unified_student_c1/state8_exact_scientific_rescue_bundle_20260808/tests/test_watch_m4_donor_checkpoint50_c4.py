from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchM4DonorCheckpoint50C4Test(unittest.TestCase):
    def test_monitor_is_frozen_recoverable_and_fail_closed(self) -> None:
        text = (ROOT / "scripts/watch_m4_donor_checkpoint50_c4.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "CHECKPOINT_ROUND",
            "checkpoint${checkpoint_round}-c4",
            "validate_checkpoint_c4_probe_amlt_config.py",
            '--round "$checkpoint_round"',
            "--status pass",
            "raw_history_visible_to_decoder",
            "persistent_history_kv_cache",
            "all_expected_jobs_pass",
            "recovery in 0 1 2 3",
        ):
            self.assertIn(required, text)
        self.assertNotIn("train_recurrent_student_m4.py", text)


if __name__ == "__main__":
    unittest.main()
