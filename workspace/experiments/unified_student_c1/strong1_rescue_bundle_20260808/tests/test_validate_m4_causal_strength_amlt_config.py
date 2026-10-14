from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_m4_causal_strength_amlt_config import load_config, validate  # noqa: E402


CONFIG = ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"


class M4CausalStrengthConfigTest(unittest.TestCase):
    def test_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_selected_job_subset_passes(self) -> None:
        config = load_config(CONFIG)
        selected = {config["jobs"][0]["name"]}
        config["jobs"] = [config["jobs"][0]]
        validate(config, selected)

    def test_branch_and_checkpoint_drift_fail(self) -> None:
        config = load_config(CONFIG)
        branch = copy.deepcopy(config)
        branch["jobs"][0]["submit_args"]["env"]["CAUSAL_WEIGHT"] = "1.0"
        with self.assertRaisesRegex(ValueError, "Branch label drift"):
            validate(branch)
        checkpoint = copy.deepcopy(config)
        checkpoint["jobs"][0]["command"] = checkpoint["jobs"][0]["command"].replace(
            "checkpoint-round-250", "checkpoint-round-251"
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(checkpoint)


if __name__ == "__main__":
    unittest.main()
