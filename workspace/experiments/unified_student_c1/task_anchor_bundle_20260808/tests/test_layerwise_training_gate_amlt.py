from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_layerwise_training_gate_amlt import load, validate  # noqa: E402


class LayerwiseTrainingGateAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_training_gate_20260731.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_scalar_state_flag_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] += "\n--init-recurrent-state bad.pt\n"
        with self.assertRaisesRegex(ValueError, "scalar"):
            validate(value)

    def test_wrong_gpu_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)

    def test_normalized_dump_target_passes(self) -> None:
        value = copy.deepcopy(self.config)
        value["target"] = {
            "name": "Feeds",
            "service": "sing",
            "workspace_name": "CS-NewsAndFeeds-Singularity",
            "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
        }
        validate(value)


if __name__ == "__main__":
    unittest.main()
