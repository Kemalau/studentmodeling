from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from summarize_state_swap_causal_effect import summarize  # noqa: E402


class StateSwapCausalEffectTest(unittest.TestCase):
    def test_clean_effect_is_relative_to_paired_controls(self) -> None:
        records = []
        categories = {
            1: {"none": "stay_B", "noop": "stay_B", "random": "stay_B", "samemal": "stay_B", "swap": "flip_to_A"},
            2: {"none": "flip_to_A", "noop": "flip_to_A", "random": "flip_to_A", "samemal": "flip_to_A", "swap": "flip_to_A"},
        }
        for pair_idx, conditions in categories.items():
            for condition, category in conditions.items():
                records.append(
                    {
                        "layer_config": "all",
                        "pair_idx": pair_idx,
                        "condition": condition,
                        "category": category,
                        "p_A": 0.9 if category == "flip_to_A" else 0.1,
                        "score_A": 1.0 if category == "flip_to_A" else 0.0,
                        "score_B": 0.0 if category == "flip_to_A" else 1.0,
                    }
                )
        result = summarize({"records": records})["all"]
        self.assertEqual(result["n_clean_none_stay_B"], 1)
        self.assertEqual(result["clean_none_stay_B"]["swap_flip_to_A_rate"], 1.0)
        self.assertEqual(result["clean_none_stay_B"]["swap_minus_noop_flip_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
