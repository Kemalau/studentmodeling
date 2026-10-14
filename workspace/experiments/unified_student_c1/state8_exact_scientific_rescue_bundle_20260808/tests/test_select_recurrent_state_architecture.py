from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from select_recurrent_state_architecture import select  # noqa: E402


def row(decoder=None, passed=True):
    value = {
        "D_specific": 0.2 if passed else 0.0,
        "clean_n": 60,
        "noop_flip_to_A": 0.0,
        "same_rule_flip_to_A": 0.0,
        "scientific_gate_pass": passed,
    }
    if decoder is not None:
        value["decoder"] = decoder
    return value


class RecurrentStateArchitectureSelectionTest(unittest.TestCase):
    def setUp(self):
        self.scalar = {"rows": [{"pass": False}, {"pass": False}]}
        self.bridge = {"rows": [row()]}
        self.cross = {"rows": [row("full_task")]}

    def test_zero_training_full_task_recurrence_precedes_training(self):
        result = select(
            self.scalar,
            self.bridge,
            self.cross,
            {"rows": [row("full_task")]},
        )
        self.assertEqual(
            result["decision"], "formal_eval_layerwise_recurrent_full_task"
        )
        self.assertTrue(result["full_unified_fit_authorized"])

    def test_bridge_without_recurrence_trains_transition_only(self):
        result = select(
            self.scalar,
            self.bridge,
            self.cross,
            {"rows": [row("full_task", False)]},
        )
        self.assertEqual(result["decision"], "train_layerwise_transition_only")
        self.assertFalse(result["full_unified_fit_authorized"])

    def test_failed_bridge_blocks_full_training(self):
        result = select(
            self.scalar,
            {"rows": [row(passed=False)]},
            {"rows": [row("full_task", False)]},
            {"rows": [row("full_task", False)]},
        )
        self.assertEqual(result["decision"], "continue_causal_objective_diagnosis")


if __name__ == "__main__":
    unittest.main()
