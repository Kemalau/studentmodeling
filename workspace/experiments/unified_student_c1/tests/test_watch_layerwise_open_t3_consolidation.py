from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/watch_layerwise_open_t3_consolidation.sh"


class OpenT3ConsolidationWatcherTest(unittest.TestCase):
    def test_watcher_recovers_each_arm_and_runs_all_gates(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('submit_recovery "$job"', source)
        self.assertIn("waves[\"$job\"] <= 6", source)
        self.assertIn("judge_formal_t3_predictions.py", source)
        self.assertIn("summarize_t3_seen_unseen.py", source)
        self.assertIn("summarize_layerwise_single_task_retention.py", source)
        self.assertIn("summarize_layerwise_open_t3_consolidation.py", source)
        self.assertIn("scientific_gate_pass", source)
        self.assertIn("record_layerwise_c4_pruned_candidate.py", source)
        self.assertIn("frozen_full_c4_specificity_gate_failed", source)
        self.assertIn("pruned_receipts", source)


if __name__ == "__main__":
    unittest.main()
