from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m4_causal_bootstrap_g1_sensitivity import load, render  # noqa: E402
from validate_m4_causal_bootstrap_g1_sensitivity import validate  # noqa: E402


class G1BootstrapSensitivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.value = render(
            load(ROOT / "amlt/feeds_m4_causal_bootstrap_rehearsal_selected_20260731.yaml")
        )

    def test_rendered_sensitivity_is_isolated_and_batch_matched(self) -> None:
        validate(self.value)
        job = self.value["jobs"][0]
        command = str(job["command"])
        self.assertEqual(job["sku"], "80G1-A100")
        self.assertIn("--diagnostic-max-micro-steps 2000", command)
        self.assertIn("--gradient-accumulation-steps 4", command)
        self.assertIn("--causal-bootstrap-fraction 0.25", command)
        self.assertIn("optimizer_steps')) == 500", command)
        self.assertNotIn("--distributed-mpi", command)

    def test_topology_or_output_collision_fails(self) -> None:
        drift = copy.deepcopy(self.value)
        drift["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU/UAI/branch drift"):
            validate(drift)
        collision = copy.deepcopy(self.value)
        collision["jobs"][0]["command"] = str(collision["jobs"][0]["command"]).replace(
            "world_model_v1_m4_causal_bootstrap_sum_w1_g1_sensitivity_seed42_20260731",
            "world_model_v1_m4_causal_bootstrap_sum_w1_seed42_round500_20260731",
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(collision)


if __name__ == "__main__":
    unittest.main()
