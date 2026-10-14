from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_layerwise_eedi_state_path_ab.sh"


class EediStatePathWatcherTest(unittest.TestCase):
    def test_watcher_recovers_siblings_and_runs_all_gates(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        for fragment in (
            "terminal_nonpass_job_names",
            "validate_layerwise_eedi_state_path_ab_amlt.py",
            "feeds_submission_global.lock",
            "--status pass",
            "judge_formal_t3_predictions.py",
            "summarize_layerwise_eedi_state_path_ab.py",
            "eval/t6_full",
            "eval/c4",
        ):
            self.assertIn(fragment, source)


if __name__ == "__main__":
    unittest.main()
