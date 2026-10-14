from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_open_t3_consolidation_amlt import load, validate  # noqa: E402


class OpenT3ConsolidationAMLTTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_open_t3_consolidation_20260803.yaml")

    def test_valid_config(self) -> None:
        validate(self.config)

    def test_rejects_wrong_phase_boundary(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["submit_args"]["env"]["DIAGNOSTIC_START_GLOBAL_POSITION"] = "12000"
        with self.assertRaises(ValueError):
            validate(value)

    def test_rejects_shared_warm_output(self) -> None:
        value = copy.deepcopy(self.config)
        first = value["jobs"][0]["submit_args"]["env"]["WARM_START_CHECKPOINT_DIR"]
        value["jobs"][1]["submit_args"]["env"]["WARM_START_CHECKPOINT_DIR"] = first
        with self.assertRaises(ValueError):
            validate(value)

    def test_accepts_amlt_dump_code_root(self) -> None:
        value = copy.deepcopy(self.config)
        value["code"].pop("local_dir")
        value["code"]["storage_id"] = "default"
        validate(value)

    def test_rejects_unrecognized_dump_code_root(self) -> None:
        value = copy.deepcopy(self.config)
        value["code"].pop("local_dir")
        value["code"]["storage_id"] = "other"
        with self.assertRaises(ValueError):
            validate(value)


if __name__ == "__main__":
    unittest.main()
