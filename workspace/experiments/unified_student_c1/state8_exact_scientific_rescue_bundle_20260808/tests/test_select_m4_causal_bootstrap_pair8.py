from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_m4_causal_bootstrap_pair8 import select  # noqa: E402


def gate(branch: str, native: float, donor: float, flip: float, passed: bool = True) -> dict:
    return {
        "stage": "M4_causal_pair8_bootstrap",
        "pass": passed,
        "checks": {"causal": passed},
        "margin_drop": 0.2,
        "state_relative_delta_gain": 0.1,
        "summary": {
            "branch": branch,
            "causal_validation": {
                "n": 8,
                "clean_n": 6,
                "native_accuracy": native,
                "donor_accuracy": donor,
                "directed_flip_to_A_on_clean": flip,
            },
        },
    }


class Pair8CurriculumSelectionTest(unittest.TestCase):
    def test_prefers_native_retention_before_tie_breaking_on_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            donor = root / "donor.json"
            anchored = root / "native25.json"
            donor.write_text(json.dumps(gate("pair8_sum_bootstrap", 0.50, 1.0, 1.0)))
            anchored.write_text(
                json.dumps(gate("pair8_sum_bootstrap_native25", 0.875, 0.75, 0.8))
            )
            result = select([("donor_only", donor), ("native25", anchored)])
        self.assertTrue(result["pass"])
        self.assertEqual(result["selected_variant"], "native25")

    def test_ignores_a_failed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            donor = root / "donor.json"
            anchored = root / "native25.json"
            donor.write_text(json.dumps(gate("pair8_sum_bootstrap", 0.75, 0.75, 0.8)))
            anchored.write_text(
                json.dumps(
                    gate("pair8_sum_bootstrap_native25", 1.0, 1.0, 1.0, passed=False)
                )
            )
            result = select([("donor_only", donor), ("native25", anchored)])
        self.assertEqual(result["selected_variant"], "donor_only")

    def test_balances_native_retention_and_donor_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            donor = root / "donor.json"
            anchored = root / "native25.json"
            donor.write_text(json.dumps(gate("pair8_sum_bootstrap", 0.875, 0.875, 1.0)))
            anchored.write_text(
                json.dumps(gate("pair8_sum_bootstrap_native25", 1.0, 0.50, 1.0))
            )
            result = select([("donor_only", donor), ("native25", anchored)])
        self.assertEqual(result["selected_variant"], "donor_only")
        self.assertEqual(result["selected"]["balanced_accuracy"], 0.875)

    def test_missing_infrastructure_candidate_does_not_block_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            donor = root / "donor.json"
            donor.write_text(json.dumps(gate("pair8_sum_bootstrap", 0.875, 1.0, 1.0)))
            result = select(
                [("donor_only", donor), ("native25", root / "missing.json")]
            )
        self.assertTrue(result["pass"])
        self.assertEqual(result["selected_variant"], "donor_only")
        self.assertEqual(result["candidates"][1]["failure"], "missing_gate")


if __name__ == "__main__":
    unittest.main()
