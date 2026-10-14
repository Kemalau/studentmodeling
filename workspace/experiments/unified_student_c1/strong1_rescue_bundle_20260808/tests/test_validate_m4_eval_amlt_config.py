from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_m4_eval_amlt_config import load_config, validate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_recurrent_student_m4_eval_20260730.yaml"


class M4EvalConfigTest(unittest.TestCase):
    def test_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_task_and_uai_drift_fail(self) -> None:
        config = load_config(CONFIG)
        task = copy.deepcopy(config)
        task["jobs"][0]["submit_args"]["env"]["TASKS"] = "predict_response"
        with self.assertRaisesRegex(ValueError, "Task label drift"):
            validate(task)
        uai = copy.deepcopy(config)
        uai["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "UAI drift"):
            validate(uai)

    def test_evaluator_hash_drift_fails(self) -> None:
        config = load_config(CONFIG)
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            "eb0243cf232d7e634f0ad784d30e66f93f42f84833a748ec453b648cff0eccda",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(config)


if __name__ == "__main__":
    unittest.main()
