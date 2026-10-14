from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_layerwise_trajectory_r64_validation_recovery_amlt import load, validate


class R64ValidationRecoveryAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_trajectory_r64_validation_recovery_20260801.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_distributed_retraining_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] += "\npython train.py --distributed-mpi\n"
        with self.assertRaisesRegex(ValueError, "Distributed"):
            validate(value)

    def test_destructive_run_dir_removal_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] += '\nrm -rf "$$RUN_DIR"\n'
        with self.assertRaisesRegex(ValueError, "destructive"):
            validate(value)

    def test_checkpoint_only_flag_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "--validation-only-from-distributed-checkpoint", ""
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
