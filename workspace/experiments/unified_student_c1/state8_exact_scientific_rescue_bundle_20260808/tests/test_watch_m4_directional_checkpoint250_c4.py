from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DirectionalCheckpoint250WatcherTest(unittest.TestCase):
    def test_watcher_is_parallel_recoverable_and_diagnostic_only(self) -> None:
        text = (ROOT / "scripts/watch_m4_directional_checkpoint250_c4.sh").read_text()
        for required in (
            "bootstrap_balanced_sum_w1",
            "bootstrap_balanced_sum_w4",
            "checkpoint-round-250/checkpoint_completion.json",
            "render_checkpoint_c4_probe.py",
            "validate_checkpoint_c4_probe_amlt_config.py",
            "for recovery in 0 1 2 3",
            "feeds_submission_global.lock",
            "--status pass",
            "scientific_gate_pass",
        ):
            self.assertIn(required, text)
        self.assertNotIn("selected_seed42.json", text)
        self.assertNotIn("pilot_complete", text)


if __name__ == "__main__":
    unittest.main()
