from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_m4_causal_pair_overfit_amlt_config import load, validate  # noqa: E402


class PairOverfitAMLTConfigTest(unittest.TestCase):
    def test_generated_filled_config_passes(self) -> None:
        validate(load(ROOT / "amlt/feeds_m4_causal_pair_overfit_20260731.yaml"))

    def test_gpu_and_reduction_drift_fail(self) -> None:
        value = load(ROOT / "amlt/feeds_m4_causal_pair_overfit_20260731.yaml")
        gpu = copy.deepcopy(value)
        gpu["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU/MPI drift"):
            validate(gpu)
        reduction = copy.deepcopy(value)
        reduction["jobs"][1]["submit_args"]["env"]["PREFERENCE_REDUCTION"] = "mean"
        with self.assertRaisesRegex(ValueError, "Branch/reduction drift"):
            validate(reduction)


if __name__ == "__main__":
    unittest.main()
