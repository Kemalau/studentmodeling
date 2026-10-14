from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WatcherTest(unittest.TestCase):
    def test_watcher_is_fail_closed_and_supervised(self):
        watcher = (ROOT / "scripts/watch_m2_m3_c4_transition.sh").read_text()
        supervisor = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text()
        self.assertIn("all_expected_jobs_pass", watcher)
        self.assertIn("terminal_nonpass_job_names", watcher)
        self.assertIn("validate_m2_m3_c4_transition_amlt.py", watcher)
        self.assertIn("raw_history_visible_to_decoder", watcher)
        self.assertIn("persistent_history_kv_cache", watcher)
        self.assertIn("m2_m3_c4_transition", supervisor)


if __name__ == "__main__":
    unittest.main()
