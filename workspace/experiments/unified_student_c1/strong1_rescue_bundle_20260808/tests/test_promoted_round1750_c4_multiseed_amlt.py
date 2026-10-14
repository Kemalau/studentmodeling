from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_promoted_round1750_c4_multiseed_amlt import load, validate  # noqa: E402


class PromotedRound1750C4MultiseedAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_promoted_round1750_c4_n150_multiseed_20260804.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_jobs_share_one_controlled_command(self) -> None:
        commands = [job["command"] for job in self.config["jobs"]]
        self.assertEqual(len(set(commands)), 1)

    def test_frozen_checkpoint_and_separate_progress_prefixes(self) -> None:
        command = self.config["jobs"][0]["command"]
        self.assertIn(
            "/mnt/amulet/studentmodeling/eval_artifacts/"
            "promoted_round1750_frozen_v1/checkpoint-round-1750",
            command,
        )
        self.assertIn(
            "/mnt/amulet/studentmodeling/eval_progress/"
            "promoted_round1750_frozen_v1/",
            command,
        )
        self.assertIn("FAILED line=%s rc=%s command=%s", command)
        self.assertNotIn(
            "/mnt/amulet/studentmodeling/checkpoints/"
            "layerwise_unified_world_model_seed42_taskretention_"
            "eedi_full_open_t3_state_path_v1_20260802/checkpoint-round-1750",
            command,
        )

    def test_single_failed_sibling_subset_passes(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"] = [config["jobs"][0]]
        validate(config, {"promoted-round1750-c4-n150-seed13"})

    def test_unexpected_recovery_job_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"] = [config["jobs"][0]]
        with self.assertRaisesRegex(ValueError, "Job-set drift"):
            validate(config, {"unknown-job"})

    def test_seed_or_manifest_drift_is_rejected(self) -> None:
        for key, value in (("PAIR_SEED", "42"), ("PAIR_MANIFEST_SHA256", "bad")):
            config = copy.deepcopy(self.config)
            config["jobs"][0]["submit_args"]["env"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "manifest drift"
            ):
                validate(config)

    def test_checkpoint_or_pair_count_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        for job in config["jobs"]:
            job["command"] = job["command"].replace(
                "checkpoint-round-1750", "checkpoint-round-1000"
            )
        with self.assertRaisesRegex(ValueError, "wrong-checkpoint"):
            validate(config)
        config = copy.deepcopy(self.config)
        for job in config["jobs"]:
            job["command"] = job["command"].replace(
                "--num-pairs 150", "--num-pairs 50"
            )
        with self.assertRaisesRegex(ValueError, "n50"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
