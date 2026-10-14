from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/summarize_recurrent_factorial.py"


class RecurrentFactorialSummaryTest(unittest.TestCase):
    def test_failed_strict_gate_authorizes_only_supported_u4_rescue(self) -> None:
        jobs = {
            "recurrent-u1-identity-seed42": (1, "identity", -0.19),
            "recurrent-u1-trainable-seed42": (1, "trainable", -0.18),
            "recurrent-u4-identity-seed42": (4, "identity", -0.16),
            "recurrent-u4-trainable-seed42": (4, "trainable", -0.11),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for job, (unroll, projection, gap) in jobs.items():
                target = root / job / "metrics/m2_gate.json"
                target.parent.mkdir(parents=True)
                target.write_text(
                    json.dumps(
                        {
                            "unroll": unroll,
                            "projection_mode": projection,
                            "scientific_gate_pass": False,
                            "checks": {
                                "candidate_support": True,
                                "direct_task_sufficiency": True,
                                "accuracy_gap": False,
                                "nonconstant_variance": True,
                                "nonconstant_cosine": True,
                            },
                            "metrics": {
                                "recurrent_accuracy": 0.75,
                                "direct_accuracy": 0.90,
                                "accuracy_gap_recurrent_minus_direct": gap,
                                "gold_nll_gap_recurrent_minus_direct": 0.03,
                                "recurrent_direct_cosine": 0.95,
                                "state_variance": 0.1,
                                "mean_pairwise_cosine": 0.96,
                                "candidate_n": 100,
                            },
                        }
                    )
                )
            output = root / "summary.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input-root",
                    str(root),
                    "--output-json",
                    str(output),
                    "--output-md",
                    str(root / "summary.md"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            value = json.loads(output.read_text())
            self.assertFalse(value["scientific_gate_pass"])
            self.assertTrue(value["downstream_rescue_authorized"])
            self.assertEqual(value["selection_mode"], "m3_behavioral_rescue")
            self.assertEqual(value["selected_projection_mode"], "trainable")


if __name__ == "__main__":
    unittest.main()
