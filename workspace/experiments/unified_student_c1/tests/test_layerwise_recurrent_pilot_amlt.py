from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_layerwise_recurrent_pilot_amlt import load, validate  # noqa: E402


class LayerwiseRecurrentPilotAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_recurrent_pilot_20260731.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_scalar_recurrent_state_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] += "\n--init-recurrent-state bad.pt\n"
        with self.assertRaisesRegex(ValueError, "scalar"):
            validate(value)

    def test_branch_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][1]["submit_args"]["env"]["LEARNING_RATE"] = "1e-3"
        with self.assertRaisesRegex(ValueError, "Branch env"):
            validate(value)

    def test_wrong_gpu_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][2]["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)

    def test_rank_local_gpu_isolation_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            'export LOCAL_RANK=0\n', ""
        )
        with self.assertRaisesRegex(ValueError, "controlled command|command drift"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
