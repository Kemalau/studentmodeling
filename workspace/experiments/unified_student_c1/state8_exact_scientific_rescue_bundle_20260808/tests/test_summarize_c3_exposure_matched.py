from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_c3_exposure_matched import transfer_conclusion  # noqa: E402


def row(eedi: float, overall: float, *, passed: bool = True) -> dict:
    return {
        "all_gates_pass": passed,
        "non_T3_preservation_pass": True,
        "T3_semantic": {
            "eedi_accuracy": eedi,
            "semantic_accuracy": overall,
            "pass": passed,
        },
        "C4": {"D_specific": 0.20},
    }


class C3ConclusionTest(unittest.TestCase):
    def test_positive_effect_recommends_replication_not_promotion(self) -> None:
        value = transfer_conclusion(
            {
                "malrule_correctness": row(0.20, 0.36),
                "malrule_response": row(0.24, 0.38),
                "malrule_misconception": row(0.32, 0.41),
            }
        )
        self.assertTrue(value["C3_transfer_hypothesis_supported"])
        self.assertTrue(value["replication_recommended"])
        self.assertFalse(value["recipe_promotion_allowed"])
        self.assertAlmostEqual(
            value["misconception_minus_best_control_real_eedi_T3"], 0.08
        )

    def test_preservation_failure_blocks_positive_claim(self) -> None:
        rows = {
            "malrule_correctness": row(0.20, 0.36),
            "malrule_response": row(0.24, 0.38),
            "malrule_misconception": row(0.40, 0.44),
        }
        rows["malrule_misconception"]["non_T3_preservation_pass"] = False
        value = transfer_conclusion(rows)
        self.assertFalse(value["C3_transfer_hypothesis_supported"])
        self.assertFalse(value["replication_recommended"])

    def test_negative_controls_need_not_pass_t3_gate(self) -> None:
        rows = {
            "malrule_correctness": row(0.08, 0.28, passed=False),
            "malrule_response": row(0.12, 0.31, passed=False),
            "malrule_misconception": row(0.32, 0.41),
        }
        value = transfer_conclusion(rows)
        self.assertTrue(value["C3_transfer_hypothesis_supported"])


if __name__ == "__main__":
    unittest.main()
