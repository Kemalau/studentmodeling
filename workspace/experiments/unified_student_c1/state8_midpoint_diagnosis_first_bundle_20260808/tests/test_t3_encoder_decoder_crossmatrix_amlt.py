from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_t3_encoder_decoder_crossmatrix_amlt import load, validate  # noqa: E402


class T3EncoderDecoderCrossmatrixAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_t3_encoder_decoder_crossmatrix_20260804.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_partial_panel_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] += "\necho --max-samples-per-group\n"
        with self.assertRaisesRegex(ValueError, "partial"):
            validate(config)

    def test_decoder_hash_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "31db2cbe6163e2a32ffe71688803d75b707b7624f79af4a6a1f4a6892138835a",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(config)

    def test_recovery_watcher_uses_the_generic_full_t3_controller(self) -> None:
        source = (
            ROOT / "scripts/watch_current_lr2e6_fulltask_crossadapter_full_t3.sh"
        ).read_text()
        self.assertIn("watch_full_t3_experiment.sh", source)
        self.assertIn("validate_t3_encoder_decoder_crossmatrix_amlt.py", source)
        self.assertIn(
            "current-lr2e6-fulltask-crossadapter-full-t3-seed42-20260804",
            source,
        )


if __name__ == "__main__":
    unittest.main()
