from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_state8_promotion_eval_amlt import validate  # noqa: E402


class State8PromotionEvalAmltTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "amlt/feeds_state8_promotion_eval_20260805.yaml").read_text()
        )

    def test_frozen_config_passes(self) -> None:
        validate(copy.deepcopy(self.config))
        self.assertEqual(len(self.config["jobs"]), 21)

    def test_rejects_non_single_gpu_job(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "Topology drift"):
            validate(value)

    def test_rejects_duplicate_shard_coordinate(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][1]["submit_args"]["env"]["SHARD_INDEX"] = "0"
        with self.assertRaisesRegex(ValueError, "T3 shard drift"):
            validate(value)

    def test_rejects_checkpoint_variant_drift(self) -> None:
        value = copy.deepcopy(self.config)
        balanced = next(job for job in value["jobs"] if job["name"].startswith("state8-balanced-"))
        balanced["submit_args"]["env"]["PROMOTION_VARIANT"] = "t3_heavy"
        with self.assertRaisesRegex(ValueError, "variant drift"):
            validate(value)

    def test_t3_jobs_use_remote_resumable_generation_progress(self) -> None:
        command = self.config["jobs"][0]["command"]
        self.assertIn("--generation-progress-dir", command)
        self.assertIn("--generation-progress-key", command)
        self.assertIn("/mnt/amulet/studentmodeling/eval_progress/", command)


if __name__ == "__main__":
    unittest.main()
