from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_recurrent_c4_robustness import summarize  # noqa: E402


def record(pair: int, condition: str, category: str, margin: float, p_a: float) -> dict:
    return {
        "pair_idx": pair,
        "condition": condition,
        "category": category,
        "margin_A_minus_B": margin,
        "p_A": p_a,
    }


class RecurrentC4RobustnessTest(unittest.TestCase):
    def test_paired_effect_matches_frozen_d_specific(self) -> None:
        records = []
        values = {
            1: {"native": ("stay_B", -2), "noop": ("stay_B", -2), "swap": ("flip_to_A", 2), "random": ("stay_B", -1), "samemal": ("stay_B", -2)},
            2: {"native": ("stay_B", -1), "noop": ("stay_B", -1), "swap": ("stay_B", -0.5), "random": ("stay_B", -0.8), "samemal": ("stay_B", -1)},
            3: {"native": ("other", -1), "noop": ("other", -1), "swap": ("flip_to_A", 1), "random": ("stay_B", -1), "samemal": ("stay_B", -1)},
        }
        for pair_idx, conditions in values.items():
            for condition, (category, margin) in conditions.items():
                records.append(record(pair_idx, condition, category, margin, 0.5))
        result = summarize(
            {"records": records, "summary": {"D_specific": 0.5}},
            draws=200,
            seed=7,
        )
        self.assertEqual(result["clean_n"], 2)
        self.assertEqual(
            result["paired_bootstrap"]["D_specific_paired_flip"]["mean"], 0.5
        )
        self.assertTrue(result["does_not_replace_scientific_gate"])

    def test_incomplete_pair_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize(
                {
                    "records": [record(1, "native", "stay_B", -1, 0.1)],
                    "summary": {"D_specific": 0},
                },
                draws=10,
            )


if __name__ == "__main__":
    unittest.main()
