from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WatchM4CF4SingleHedgeTest(unittest.TestCase):
    def test_watcher_is_recoverable_and_fail_closed(self) -> None:
        text = (ROOT / "scripts/watch_m4_cf4_single_hedge.sh").read_text()
        for value in (
            "feeds_submission_global.lock",
            "for recovery in 0 1 2 3",
            "validate_m4_cf4_single_hedge_amlt_config.py",
            "all_expected_jobs_pass",
            "raw_history_visible_to_decoder",
            "persistent_history_kv_cache",
            "completion['micro_steps'] == 1000",
            "completion['optimizer_steps'] == 250",
            "C4_D_specific",
            "screen_only",
            "scientific_gate_pass",
            "scientific_gate_failed",
            'touch "$state_dir/complete"',
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
