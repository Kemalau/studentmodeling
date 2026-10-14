from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_expanded_head_backward_gate_amlt import load, validate  # noqa: E402


class ExpandedHeadBackwardGateAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_layerwise_expanded_head_backward_gate_20260802.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_exact_row_mapping_is_frozen(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][1]["submit_args"]["env"]["ROW_INDEX"] = 110500
        with self.assertRaisesRegex(ValueError, "mapping drift"):
            validate(value)

    def test_single_a100_and_uai_are_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "UAI"):
            validate(value)

    def test_all_jobs_share_the_frozen_launcher(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][2]["command"] += "\necho drift\n"
        with self.assertRaisesRegex(ValueError, "share one command"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
