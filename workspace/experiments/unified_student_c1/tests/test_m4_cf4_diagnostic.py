from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m4_cf4_diagnostic import SPECS, load, render  # noqa: E402
from validate_m4_cf4_diagnostic_amlt_config import validate  # noqa: E402


class M4CF4DiagnosticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.value = render(
            load(ROOT / "amlt/feeds_m4_causal_donor_directional_preemptible_20260731.yaml")
        )

    def test_rendered_config_passes(self) -> None:
        validate(self.value)
        self.assertEqual(
            {job["name"] for job in self.value["jobs"]},
            {spec["job"] for spec in SPECS.values()},
        )

    def test_curricula_share_data_weight_and_horizon(self) -> None:
        for job in self.value["jobs"]:
            command = str(job["command"])
            self.assertIn("cf4_balanced_paired_state_swap_train", command)
            self.assertIn("--diagnostic-stop-after-round 250", command)
            self.assertEqual(job["submit_args"]["env"]["CAUSAL_WEIGHT"], "1.0")
            self.assertIn("export CUDA_VISIBLE_DEVICES=0", command)

    def test_data_or_gpu_drift_fails(self) -> None:
        drift = copy.deepcopy(self.value)
        drift["jobs"][0]["command"] = str(drift["jobs"][0]["command"]).replace(
            "cf4_balanced_paired_state_swap_train", "cf_paired_state_swap_train"
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(drift)

    def test_single_failed_job_subset_can_be_validated_for_recovery(self) -> None:
        subset = copy.deepcopy(self.value)
        subset["jobs"] = [subset["jobs"][1]]
        name = subset["jobs"][0]["name"]
        validate(subset, {name})


if __name__ == "__main__":
    unittest.main()
