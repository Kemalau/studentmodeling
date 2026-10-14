from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_layerwise_recurrent_causal_gate import categorize  # noqa: E402


class EvaluateLayerwiseRecurrentCausalGateTest(unittest.TestCase):
    def test_categorize(self) -> None:
        self.assertEqual(categorize("A", "A", "B"), "flip_to_A")
        self.assertEqual(categorize("B", "A", "B"), "stay_B")
        self.assertEqual(categorize("C", "A", "B"), "other")

    def test_general_recurrent_gate_accepts_layerwise_architecture(self) -> None:
        source = (SCRIPT_DIR / "evaluate_recurrent_causal_gate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('choices=("scalar", "layerwise")', source)
        self.assertIn('"final_layerwise_recurrent_state"', source)


if __name__ == "__main__":
    unittest.main()
