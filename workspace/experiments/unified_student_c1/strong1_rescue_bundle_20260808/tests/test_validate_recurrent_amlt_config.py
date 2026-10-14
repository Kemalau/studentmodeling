from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_recurrent_amlt_config import load_config, validate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_recurrent_student_state_u1_u4_factorial_20260730.yaml"


class RecurrentAMLTConfigTest(unittest.TestCase):
    def test_frozen_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_target_and_uai_drift_fail(self) -> None:
        config = load_config(CONFIG)
        target_drift = copy.deepcopy(config)
        target_drift["target"]["name"] = "Other"
        with self.assertRaisesRegex(ValueError, "Target drift"):
            validate(target_drift)
        uai_drift = copy.deepcopy(config)
        uai_drift["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "UAI drift"):
            validate(uai_drift)

    def test_prefixed_dump_can_be_loaded(self) -> None:
        temporary = ROOT / "tests/.tmp_recurrent_amlt_dump.yaml"
        try:
            temporary.write_text("AMLT info line\n" + CONFIG.read_text(encoding="utf-8"))
            validate(load_config(temporary))
        finally:
            temporary.unlink(missing_ok=True)

    def test_amlt_normalized_target_passes(self) -> None:
        config = load_config(CONFIG)
        config["target"] = {
            "name": "Feeds",
            "workspace_name": "CS-NewsAndFeeds-Singularity",
            "workspace_resource_group": "rg-cs-newsandfeeds-singularity",
        }
        validate(config)

    def test_nearby_normalized_target_drift_fails(self) -> None:
        config = load_config(CONFIG)
        config["target"] = {
            "name": "Feeds",
            "workspace_name": "CS-NewsAndFeeds-Singularity",
            "workspace_resource_group": "other-rg",
        }
        with self.assertRaisesRegex(ValueError, "Target drift"):
            validate(config)

    def test_recurrent_model_hash_drift_fails(self) -> None:
        config = load_config(CONFIG)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "9e0431dccb524f67b401f7ea911e11c96fc3061432ffdc73e0cac2bbc2acc5a0",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Training command drift"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
