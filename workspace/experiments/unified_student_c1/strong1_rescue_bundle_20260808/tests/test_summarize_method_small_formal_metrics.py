from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_method_small_formal_metrics import summarize  # noqa: E402


class MethodSmallFormalSummaryTest(unittest.TestCase):
    def test_weighted_support_not_unweighted_group_mean(self) -> None:
        payload = {
            "metrics_by_group": {
                "predict_correctness/a/protocol": {"n": 1, "accuracy": 1.0},
                "predict_correctness/b/protocol": {"n": 3, "accuracy": 0.0},
                "predict_response/a/protocol": {"n": 2, "math_equivalence": 0.5},
                "diagnose_misconception/a/protocol": {"n": 2, "normalized_exact_match": 0.5},
                "predict_prospective_error/a/protocol": {"n": 2, "joint_math_equivalence": 0.5},
                "predict_error_persistence/a/protocol": {"n": 2, "accuracy": 0.5},
            }
        }
        result = summarize(payload)
        self.assertEqual(result["T1"]["value"], 0.25)
        self.assertEqual(result["T1"]["n"], 4)


if __name__ == "__main__":
    unittest.main()
