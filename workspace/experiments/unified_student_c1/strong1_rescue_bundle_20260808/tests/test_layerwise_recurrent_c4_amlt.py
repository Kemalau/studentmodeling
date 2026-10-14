from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from render_layerwise_recurrent_c4 import load, render  # noqa: E402
from validate_layerwise_recurrent_c4_amlt import validate  # noqa: E402


class LayerwiseRecurrentC4AmltTest(unittest.TestCase):
    def test_three_zero_training_jobs_validate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        value = render(load(root / "amlt/feeds_one_shot_layer_transport_c4_20260731.yaml"))
        validate(value)
        self.assertEqual(len(value["jobs"]), 3)
        for job in value["jobs"]:
            self.assertIn("optimizer_steps=0", str(job["command"]))
            self.assertNotIn("train_recurrent_student", str(job["command"]))


if __name__ == "__main__":
    unittest.main()
