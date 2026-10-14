from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_selective_kd_lr2e6_full_t3_amlt import load, validate  # noqa: E402


class SelectiveKdLr2e6FullT3AmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_selective_kd_lr2e6_full_t3_20260804.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_state_hash_drift_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "d275719bb19f1000523c067cfdce2a5863b4e1c553a79e2e2cb7a030bc3abcca",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(config)

    def test_cross_adapter_is_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["jobs"][0]["command"] += "\necho --decode-adapter\n"
        with self.assertRaisesRegex(ValueError, "cross-adapter"):
            validate(config)


if __name__ == "__main__":
    unittest.main()

