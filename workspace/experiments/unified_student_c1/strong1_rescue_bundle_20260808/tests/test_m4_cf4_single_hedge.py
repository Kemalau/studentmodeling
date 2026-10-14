from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m4_cf4_single_hedge import JOB, load, render  # noqa: E402
from validate_m4_cf4_single_hedge_amlt_config import validate  # noqa: E402


class M4CF4SingleHedgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.value = render(load(ROOT / "amlt/feeds_m4_cf4_diagnostic_20260731.yaml"))

    def test_rendered_config_is_valid_and_batch_equivalent(self) -> None:
        validate(self.value)
        job = self.value["jobs"][0]
        self.assertEqual(job["name"], JOB)
        self.assertEqual(job["sku"], "80G1-A100")
        self.assertFalse(job["mpi"])
        command = job["command"]
        self.assertIn("--diagnostic-max-micro-steps 1000", command)
        self.assertIn("--gradient-accumulation-steps 4", command)
        self.assertIn("optimizer_steps') == 250", command)
        self.assertNotIn("--distributed-mpi", command)

    def test_topology_or_protocol_drift_fails(self) -> None:
        drift = copy.deepcopy(self.value)
        drift["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU/MPI/SLA drift"):
            validate(drift)


if __name__ == "__main__":
    unittest.main()
