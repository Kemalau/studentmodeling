from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_task_retention_promoted_final_validation_amlt import (  # noqa: E402
    load,
    validate,
)


class PromotedFinalValidationAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT
            / "amlt/feeds_layerwise_task_retention_promoted_final_validation_recovery_20260804.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_gpu_or_uai_drift_fails(self) -> None:
        for key, value in (("sku", "2x80G8-A100-IB-NvLink"), ("mpi", True)):
            config = copy.deepcopy(self.config)
            config["jobs"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(config)
        config = copy.deepcopy(self.config)
        config["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaises(ValueError):
            validate(config)

    def test_launcher_drift_fails(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = "set -euo pipefail\nbash scripts/wrong.sh\n"
        with self.assertRaisesRegex(ValueError, "launcher"):
            validate(config)

    def test_script_requires_exact_final_checkpoint_and_validation_only(self) -> None:
        script = (ROOT / "scripts/run_layerwise_task_retention_promoted_final_validation.sh").read_text()
        self.assertIn("checkpoint-round-16241", script)
        self.assertIn("--validation-only-from-distributed-checkpoint", script)
        self.assertIn("--finalize-complete-distributed-epoch", script)
        self.assertNotIn("--distributed-mpi", script)


if __name__ == "__main__":
    unittest.main()
