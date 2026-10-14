from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_eedi_state_path_ab_amlt import load, validate  # noqa: E402


class EediStatePathAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_eedi_state_path_ab_20260802.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_profiles_are_matched_except_for_loss_profile(self) -> None:
        jobs = self.config["jobs"]
        left = jobs[0]["submit_args"]["env"]
        right = jobs[1]["submit_args"]["env"]
        self.assertEqual(left["MANIFEST_SHA256"], right["MANIFEST_SHA256"])
        self.assertEqual({left["LOSS_PROFILE"], right["LOSS_PROFILE"]}, {"standard", "state_path"})

    def test_launcher_forces_the_job_venv_interpreter(self) -> None:
        launcher = (ROOT / "scripts/run_layerwise_task_retention_rehearsal.sh").read_text()
        for job in self.config["jobs"]:
            self.assertEqual(
                job["submit_args"]["env"]["VENV_ROOT"],
                "/tmp/studentmodeling-layerwise-eedi-ab-venv",
            )
        self.assertIn('python() { "$VENV_ROOT/bin/python" "$@"; }', launcher)
        self.assertIn("import accelerate,bitsandbytes,peft,safetensors,transformers", launcher)

    def test_profile_or_gpu_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["LOSS_PROFILE"] = "state_path"
        with self.assertRaisesRegex(ValueError, "arm drift"):
            validate(value)
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
