from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_layerwise_known_oom_row_recovery_amlt import validate  # noqa: E402


class KnownOomRowConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (ROOT / "amlt/feeds_layerwise_known_oom_row_recovery_20260801.yaml").read_text()
        )

    def test_committed_config_passes(self) -> None:
        validate(copy.deepcopy(self.config))

    def test_offload_threshold_is_frozen(self) -> None:
        drift = copy.deepcopy(self.config)
        drift["jobs"][0]["command"] = drift["jobs"][0]["command"].replace(
            "--direct-offload-min-tokens 512", "--direct-offload-min-tokens 1"
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(drift)

    def test_requires_one_a100(self) -> None:
        drift = copy.deepcopy(self.config)
        drift["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(drift)


if __name__ == "__main__":
    unittest.main()
