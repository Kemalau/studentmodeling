from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_m3_gate import arm_matches_path, select_arm  # noqa: E402


class M3GateSummaryTest(unittest.TestCase):
    def test_matches_amlt_hyphenated_and_remote_underscored_paths(self) -> None:
        self.assertTrue(
            arm_matches_path(
                "generation_only",
                Path("experiment/m3-generation-only-identity-seed42/metrics/validation_metrics.json"),
            )
        )
        self.assertTrue(
            arm_matches_path(
                "generation_kd",
                Path("world_model_v1_m3_generation_kd_identity/validation_metrics.json"),
            )
        )
        self.assertFalse(
            arm_matches_path(
                "generation_kd",
                Path("experiment/m3-generation-only-identity/metrics/validation_metrics.json"),
            )
        )

    def test_strict_selection_has_priority_over_rescue(self) -> None:
        rows = {
            "generation_only": {
                "pass": True,
                "rescue_pass": True,
                "metrics": {
                    "accuracy_gap_recurrent_minus_direct": -0.09,
                    "gold_nll_gap_recurrent_minus_direct": 0.02,
                },
            },
            "generation_kd": {
                "pass": False,
                "rescue_pass": True,
                "metrics": {
                    "accuracy_gap_recurrent_minus_direct": -0.01,
                    "gold_nll_gap_recurrent_minus_direct": 0.01,
                },
            },
        }
        self.assertEqual(select_arm(rows), ("generation_only", "strict_m3"))

    def test_noncollapsed_arm_can_enter_m4_causal_rescue(self) -> None:
        rows = {
            "generation_only": {
                "pass": False,
                "rescue_pass": False,
                "metrics": {
                    "accuracy_gap_recurrent_minus_direct": -0.30,
                    "gold_nll_gap_recurrent_minus_direct": 0.03,
                },
            },
            "generation_kd": {
                "pass": False,
                "rescue_pass": True,
                "metrics": {
                    "accuracy_gap_recurrent_minus_direct": -0.12,
                    "gold_nll_gap_recurrent_minus_direct": 0.01,
                },
            },
        }
        self.assertEqual(select_arm(rows), ("generation_kd", "m4_causal_rescue"))


if __name__ == "__main__":
    unittest.main()
