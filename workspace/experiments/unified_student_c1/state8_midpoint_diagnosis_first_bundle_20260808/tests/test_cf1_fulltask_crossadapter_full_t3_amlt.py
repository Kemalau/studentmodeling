from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_cf1_fulltask_crossadapter_full_t3_amlt import load, validate  # noqa: E402


class Cf1FulltaskCrossadapterFullT3AmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_cf1_fulltask_crossadapter_full_t3_20260804.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_decoder_hash_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(config)

    def test_partial_panel_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] += "\necho --max-samples-per-group\n"
        with self.assertRaisesRegex(ValueError, "partial"):
            validate(config)


if __name__ == "__main__":
    unittest.main()

