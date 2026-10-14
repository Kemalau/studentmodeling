from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_final_eval_amlt_config import load_config, validate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_recurrent_student_final_eval_20260730.yaml"


class FinalEvalConfigTest(unittest.TestCase):
    def test_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_receipt_and_gpu_drift_fail(self) -> None:
        config = load_config(CONFIG)
        receipt = copy.deepcopy(config)
        receipt["jobs"][0]["command"] = receipt["jobs"][0]["command"].replace("final_fit_complete", "candidate_complete")
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(receipt)
        gpu = copy.deepcopy(config)
        gpu["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU/MPI drift"):
            validate(gpu)


if __name__ == "__main__":
    unittest.main()
