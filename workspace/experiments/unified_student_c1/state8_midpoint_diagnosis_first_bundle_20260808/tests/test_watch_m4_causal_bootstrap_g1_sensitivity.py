from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class G1BootstrapWatcherTest(unittest.TestCase):
    def test_watcher_recovers_and_never_promotes(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_bootstrap_g1_sensitivity.sh").read_text()
        self.assertIn("for recovery_wave in 0 1 2 3", text)
        self.assertIn("validate_m4_causal_bootstrap_g1_sensitivity.py", text)
        self.assertIn("summarize_m4_causal_strength_branches.py", text)
        self.assertIn("bootstrap_sum_w1_g1_sensitivity", text)
        self.assertNotIn("selected_seed42.json", text)
        self.assertNotIn("pilot_complete", text)


if __name__ == "__main__":
    unittest.main()
