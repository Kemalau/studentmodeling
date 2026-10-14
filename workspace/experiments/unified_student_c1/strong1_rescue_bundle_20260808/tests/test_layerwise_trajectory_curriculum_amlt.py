from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_layerwise_trajectory_curriculum_amlt import load, validate  # noqa: E402


class LayerwiseTrajectoryCurriculumAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_layerwise_trajectory_curriculum_20260731.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_trainable_decoder_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "--layerwise-transition-only", ""
        )
        with self.assertRaisesRegex(ValueError, "command"):
            validate(value)

    def test_wrong_branch_rank_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][1]["submit_args"]["env"]["TRANSITION_RANK"] = "64"
        with self.assertRaisesRegex(ValueError, "Branch env"):
            validate(value)

    def test_wrong_gpu_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][2]["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
