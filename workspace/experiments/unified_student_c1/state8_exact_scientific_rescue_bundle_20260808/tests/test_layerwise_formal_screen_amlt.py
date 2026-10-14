from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from render_layerwise_formal_screen import load, render  # noqa: E402
from validate_layerwise_formal_screen_amlt import validate  # noqa: E402


class LayerwiseFormalScreenAmltTest(unittest.TestCase):
    def test_four_read_only_jobs_validate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        value = render(load(root / "amlt/feeds_recurrent_student_m4_eval_20260730.yaml"))
        validate(value)
        self.assertEqual(len(value["jobs"]), 4)

    def test_training_or_uai_drift_fails(self) -> None:
        root = Path(__file__).resolve().parents[1]
        value = render(load(root / "amlt/feeds_recurrent_student_m4_eval_20260730.yaml"))
        training = copy.deepcopy(value)
        training["jobs"][0]["command"] += "\npython train_recurrent_student.py\n"
        with self.assertRaisesRegex(ValueError, "Training"):
            validate(training)
        uai = copy.deepcopy(value)
        uai["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "GPU/UAI/task"):
            validate(uai)


if __name__ == "__main__":
    unittest.main()
