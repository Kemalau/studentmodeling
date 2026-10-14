from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_layerwise_checkpoint750_formal_full import render  # noqa: E402
from validate_layerwise_checkpoint750_formal_full_amlt import validate  # noqa: E402


class Checkpoint750FormalFullTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = yaml.safe_load(
            (ROOT / "amlt/feeds_layerwise_unified_final_eval_20260731.yaml").read_text()
        )
        cls.config = render(base)

    def test_rendered_config_passes(self) -> None:
        validate(copy.deepcopy(self.config))

    def test_single_sibling_recovery_passes(self) -> None:
        subset = copy.deepcopy(self.config)
        subset["jobs"] = [subset["jobs"][1]]
        validate(subset, {"layerwise-checkpoint750-full-t3-seed42"})

    def test_sampling_cap_is_rejected(self) -> None:
        drift = copy.deepcopy(self.config)
        drift["jobs"][0]["command"] += "\n--max-samples-per-group 25\n"
        with self.assertRaisesRegex(ValueError, "Small"):
            validate(drift)

    def test_round_drift_is_rejected(self) -> None:
        drift = copy.deepcopy(self.config)
        drift["jobs"][0]["command"] = drift["jobs"][0]["command"].replace(
            "checkpoint-round-750", "checkpoint-round-1000"
        )
        with self.assertRaisesRegex(ValueError, "command drift|Small"):
            validate(drift)


if __name__ == "__main__":
    unittest.main()
