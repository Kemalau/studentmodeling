from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WatchM2SeededCF4ScreenTest(unittest.TestCase):
    def test_watcher_is_recoverable_and_fail_closed(self) -> None:
        text = (ROOT / "scripts/watch_m2_seeded_cf4_screen.sh").read_text()
        for value in (
            "feeds_submission_global.lock",
            'max_recovery="${MAX_RECOVERY:-6}"',
            'start_recovery="${START_RECOVERY:-0}"',
            "for ((recovery=start_recovery; recovery<=max_recovery; recovery++))",
            'validator="${VALIDATOR:-scripts/validate_m2_seeded_cf4_screen_amlt.py}"',
            'expected_objective="${EXPECTED_OBJECTIVE:-donor_directional}"',
            "all_expected_jobs_pass",
            "raw_history_visible_to_decoder",
            "persistent_history_kv_cache",
            "initialization=m2_cf1_causal_adapter_and_state",
            "completion['micro_steps'] == 1000",
            "completion['optimizer_steps'] == 250",
            "C4_D_specific",
            "scientific_gate_pass",
            "scientific_gate_failed",
            'touch "$state_dir/complete"',
        ):
            self.assertIn(value, text)

    def test_one_watcher_can_isolate_triplet_and_donor_runs(self) -> None:
        text = (ROOT / "scripts/watch_m2_seeded_cf4_screen.sh").read_text()
        for value in (
            "STATE_SLUG",
            "CONFIG_PATH",
            "BASE_EXPERIMENT",
            "EXPECTED_BRANCH",
            "EXPECTED_OBJECTIVE",
            "VALIDATOR",
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
