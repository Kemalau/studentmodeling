from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_unified_final_validation_recovery_amlt import (  # noqa: E402
    load,
    validate,
)


class FinalValidationRecoveryAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_layerwise_unified_final_validation_recovery_20260801.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_frozen_training_runtime_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "cp tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py tests/test_layerwise_recurrent_student_model.py",
            ":",
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_checkpoint_or_finalize_flag_drift_fails(self) -> None:
        for old, new in (
            ("checkpoint-round-16241", "checkpoint-round-16240"),
            ("--finalize-complete-distributed-epoch", ""),
            ("receipt.get('world_size') == 16", "receipt.get('world_size') == 8"),
        ):
            value = copy.deepcopy(self.config)
            value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(old, new)
            with self.subTest(old=old), self.assertRaises(ValueError):
                validate(value)

    def test_gpu_or_uai_drift_fails(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "G1"
        with self.assertRaises(ValueError):
            validate(value)
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"][
            "_AZUREML_SINGULARITY_JOB_UAI"
        ] = "another_identity"
        with self.assertRaises(ValueError):
            validate(value)


if __name__ == "__main__":
    unittest.main()
