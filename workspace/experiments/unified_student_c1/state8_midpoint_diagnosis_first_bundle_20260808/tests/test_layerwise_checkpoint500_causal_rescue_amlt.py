from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_layerwise_checkpoint500_causal_rescue_amlt import load, validate  # noqa: E402


class LayerwiseCheckpoint500CausalRescueAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_layerwise_checkpoint500_causal_rescue_20260801.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_arms_share_one_command(self) -> None:
        self.assertEqual(
            self.config["jobs"][0]["command"], self.config["jobs"][1]["command"]
        )

    def test_controlled_arm_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["CAUSAL_WEIGHT"] = "2.0"
        with self.assertRaisesRegex(ValueError, "arm drift"):
            validate(value)

    def test_exact_resume_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] += "\n--resume-from-checkpoint forbidden\n"
        value["jobs"][1]["command"] = value["jobs"][0]["command"]
        with self.assertRaisesRegex(ValueError, "exact-resume"):
            validate(value)

    def test_world_size_or_boundary_drift_is_rejected(self) -> None:
        for before, after in (
            ("--distributed-world-size 4", "--distributed-world-size 8"),
            ("--diagnostic-start-global-position 8000", "--diagnostic-start-global-position 0"),
            ("--diagnostic-stop-after-round 2250", "--diagnostic-stop-after-round 250"),
        ):
            value = copy.deepcopy(self.config)
            for job in value["jobs"]:
                job["command"] = job["command"].replace(before, after)
            with self.subTest(before=before), self.assertRaisesRegex(ValueError, "command drift"):
                validate(value)


if __name__ == "__main__":
    unittest.main()
