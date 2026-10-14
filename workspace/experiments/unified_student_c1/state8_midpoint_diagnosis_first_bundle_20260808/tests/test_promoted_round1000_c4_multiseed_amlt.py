from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_promoted_round1000_c4_multiseed_amlt import load, validate  # noqa: E402


class PromotedRound1000C4MultiseedAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_promoted_round1000_c4_n150_multiseed_20260804.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_jobs_share_one_controlled_command(self) -> None:
        commands = [job["command"] for job in self.config["jobs"]]
        self.assertEqual(len(set(commands)), 1)

    def test_seed_or_manifest_drift_is_rejected(self) -> None:
        for key, value in (("PAIR_SEED", "42"), ("PAIR_MANIFEST_SHA256", "bad")):
            config = copy.deepcopy(self.config)
            config["jobs"][0]["submit_args"]["env"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "manifest drift"):
                validate(config)

    def test_gpu_or_pair_count_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["sku"] = "80G2-A100"
        with self.assertRaisesRegex(ValueError, "GPU/MPI/SLA drift"):
            validate(config)
        config = copy.deepcopy(self.config)
        for job in config["jobs"]:
            job["command"] = job["command"].replace("--num-pairs 150", "--num-pairs 50")
        with self.assertRaisesRegex(ValueError, "n50 path leaked"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
