from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from render_layerwise_state_bridge_c4 import load, render  # noqa: E402
from validate_layerwise_state_bridge_amlt import validate  # noqa: E402


class LayerwiseStateBridgeAmltTest(unittest.TestCase):
    def test_rendered_config_validates_and_contains_no_training(self) -> None:
        root = Path(__file__).resolve().parents[1]
        base = load(root / "amlt/feeds_one_shot_layer_transport_c4_20260731.yaml")
        rendered = render(base)
        validate(rendered)
        self.assertEqual(len(rendered["jobs"]), 2)
        for job in rendered["jobs"]:
            command = str(job["command"])
            self.assertIn("optimizer_steps=0", command)
            self.assertNotIn("train_recurrent_student", command)

    def test_validator_rejects_wrong_uai(self) -> None:
        root = Path(__file__).resolve().parents[1]
        rendered = render(
            load(root / "amlt/feeds_one_shot_layer_transport_c4_20260731.yaml")
        )
        rendered["jobs"][0]["submit_args"]["env"][
            "_AZUREML_SINGULARITY_JOB_UAI"
        ] = "wrong"
        with self.assertRaisesRegex(ValueError, "UAI"):
            validate(rendered)


if __name__ == "__main__":
    unittest.main()
