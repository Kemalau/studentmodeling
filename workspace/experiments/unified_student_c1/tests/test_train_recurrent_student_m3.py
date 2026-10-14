from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from train_recurrent_student_m3 import summarize_training_history  # noqa: E402


class M3TrainingSummaryTest(unittest.TestCase):
    def test_reports_every_task_source_separately(self) -> None:
        records = [
            {
                "task": "diagnose_misconception",
                "source": "eedi_mining",
                "loss": 2.0,
                "generation_loss": 1.0,
                "behavior_kd_loss": 3.0,
                "consistency_loss": 4.0,
                "direct_loss": 5.0,
            },
            {
                "task": "diagnose_misconception",
                "source": "eedi_mining",
                "loss": 4.0,
                "generation_loss": 3.0,
                "behavior_kd_loss": 5.0,
                "consistency_loss": 6.0,
                "direct_loss": 7.0,
            },
            {
                "task": "predict_correctness",
                "source": "xes3g5m",
                "loss": 1.0,
                "generation_loss": 1.0,
                "behavior_kd_loss": 1.0,
                "consistency_loss": 1.0,
                "direct_loss": 1.0,
            },
        ]
        result = summarize_training_history(records)
        self.assertEqual(set(result), {"diagnose_misconception/eedi_mining", "predict_correctness/xes3g5m"})
        self.assertEqual(result["diagnose_misconception/eedi_mining"]["n"], 2)
        self.assertEqual(result["diagnose_misconception/eedi_mining"]["mean_loss"], 3.0)


if __name__ == "__main__":
    unittest.main()
