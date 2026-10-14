from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapSummaryTest(unittest.TestCase):
    def test_passes_only_after_bootstrap_and_symmetric_tail_separate_state(self) -> None:
        row = {
            "branch": "pair8_sum_bootstrap",
            "preference_reduction": "sum",
            "objective_phase_counts": {"directional": 50, "symmetric": 150},
            "margin_loss_first_50": 0.50,
            "margin_loss_last_50": 0.20,
            "state_relative_delta_first_50": 0.01,
            "state_relative_delta_last_50": 0.08,
            "causal_validation": {
                "n": 8,
                "clean_n": 5,
                "native_accuracy": 0.75,
                "donor_accuracy": 0.75,
                "directed_flip_to_A_on_clean": 0.8,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pair_overfit_summary.json").write_text(json.dumps(row))
            output = root / "gate.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_m4_causal_bootstrap_pair8.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=False,
            )
            payload = json.loads(output.read_text())
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(payload["pass"])

    def test_rejects_a_tail_stuck_at_the_symmetric_margin(self) -> None:
        row = {
            "branch": "pair8_sum_bootstrap",
            "preference_reduction": "sum",
            "objective_phase_counts": {"directional": 50, "symmetric": 150},
            "margin_loss_first_50": 0.50,
            "margin_loss_last_50": 0.50,
            "state_relative_delta_first_50": 0.01,
            "state_relative_delta_last_50": 0.08,
            "causal_validation": {
                "n": 8,
                "clean_n": 5,
                "native_accuracy": 0.75,
                "donor_accuracy": 0.75,
                "directed_flip_to_A_on_clean": 0.8,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pair_overfit_summary.json").write_text(json.dumps(row))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_m4_causal_bootstrap_pair8.py"),
                    "--root",
                    str(root),
                    "--output",
                    str(root / "gate.json"),
                ],
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)

    def test_accepts_a_balanced_directional_tail(self) -> None:
        row = {
            "branch": "pair8_sum_bootstrap_balanced",
            "preference_reduction": "sum",
            "objective_mode": "bootstrap_then_directional",
            "objective_phase_counts": {
                "directional_bootstrap": 50,
                "directional_balanced": 150,
            },
            "margin_loss_first_50": 0.50,
            "margin_loss_last_50": 0.10,
            "state_relative_delta_first_50": 0.01,
            "state_relative_delta_last_50": 0.10,
            "causal_validation": {
                "n": 8,
                "clean_n": 6,
                "native_accuracy": 0.75,
                "donor_accuracy": 0.75,
                "directed_flip_to_A_on_clean": 0.8,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pair_overfit_summary.json").write_text(json.dumps(row))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_m4_causal_bootstrap_pair8.py"),
                    "--root", str(root),
                    "--output", str(root / "gate.json"),
                    "--expected-branch", "pair8_sum_bootstrap_balanced",
                ],
                check=False,
            )
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
