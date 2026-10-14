from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_m3_amlt_config import load_config, selected_names, validate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_recurrent_student_m3_ab_20260730.yaml"


class M3AMLTConfigTest(unittest.TestCase):
    def test_frozen_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_selected_projection_dump_passes(self) -> None:
        config = load_config(CONFIG)
        config["jobs"] = [
            job for job in config["jobs"] if job["name"] in selected_names("identity")
        ]
        validate(config, projection="identity", selected_only=True)

    def test_amlt_list_command_representation_passes(self) -> None:
        config = load_config(CONFIG)
        for job in config["jobs"]:
            job["command"] = [job["command"]]
        validate(config)

    def test_mixed_projection_selection_fails(self) -> None:
        config = load_config(CONFIG)
        config["jobs"] = [config["jobs"][0], config["jobs"][-1]]
        with self.assertRaisesRegex(ValueError, "Job set drift"):
            validate(config, projection="identity", selected_only=True)

    def test_target_uai_and_remote_model_guards(self) -> None:
        config = load_config(CONFIG)
        target_drift = copy.deepcopy(config)
        target_drift["target"]["name"] = "Other"
        with self.assertRaisesRegex(ValueError, "Target drift"):
            validate(target_drift)
        uai_drift = copy.deepcopy(config)
        uai_drift["jobs"][0]["submit_args"]["env"][
            "_AZUREML_SINGULARITY_JOB_UAI"
        ] = "other"
        with self.assertRaisesRegex(ValueError, "UAI drift"):
            validate(uai_drift)
        local_model = copy.deepcopy(config)
        local_model["jobs"][0]["command"] += "\nls /home/v-yuejiali/studentmodeling/models"
        with self.assertRaisesRegex(ValueError, "local-model path"):
            validate(local_model)

    def test_recurrent_model_hash_drift_fails(self) -> None:
        config = load_config(CONFIG)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "9e0431dccb524f67b401f7ea911e11c96fc3061432ffdc73e0cac2bbc2acc5a0",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Training command drift"):
            validate(config)

    def test_rendered_shell_syntax_drift_fails(self) -> None:
        config = load_config(CONFIG)
        config["jobs"][0]["command"] += "\nif true; then\n  echo unterminated\n"
        with self.assertRaisesRegex(ValueError, "Shell syntax error"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
