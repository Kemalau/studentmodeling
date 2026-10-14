from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WatchM4CF4DiagnosticTest(unittest.TestCase):
    def test_watcher_recovers_and_reports_frozen_c4(self) -> None:
        text = (ROOT / "scripts/watch_m4_cf4_diagnostic.sh").read_text()
        for value in (
            "feeds_submission_global.lock",
            "for recovery_wave in 0 1 2 3",
            "validate_m4_cf4_diagnostic_amlt_config.py",
            "terminal_nonpass_job_names",
            'selectors+=(":$job")',
            '--jobs "${active_jobs[@]}"',
            "expected two distinct C4 jobs",
            "recurrent_causal_gate.json",
            "C4_D_specific",
            "validation_accuracy_gap",
            'touch "$state_dir/complete"',
        ):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
