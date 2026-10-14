from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from record_layerwise_c4_pruned_candidate import build  # noqa: E402


class RecordPrunedCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = {
            "stage": "layerwise_checkpoint_local_c4",
            "checkpoint_round": 3500,
            "all_pairs_n": 150,
            "clean_n": 51,
            "swap": 0.3137254901960784,
            "random": 0.3137254901960784,
            "same_rule": 0.0,
            "noop": 0.0,
            "D_specific": 0.0,
            "D_damage": 0.3137254901960784,
            "scientific_gate_pass": False,
            "checks": {"specificity": False},
        }

    def test_records_proven_specificity_failure(self) -> None:
        value = build(
            self.evidence,
            candidate_id="candidate",
            loss_profile="state_path",
            learning_rate=5e-7,
            checkpoint_round=3500,
        )
        self.assertTrue(value["pruned"])
        self.assertFalse(value["all_gates_pass"])
        self.assertEqual(value["C4"]["D_specific"], 0.0)

    def test_rejects_passing_specificity(self) -> None:
        value = copy.deepcopy(self.evidence)
        value["D_specific"] = 0.2
        with self.assertRaises(ValueError):
            build(
                value,
                candidate_id="candidate",
                loss_profile="state_path",
                learning_rate=5e-7,
                checkpoint_round=3500,
            )


if __name__ == "__main__":
    unittest.main()
