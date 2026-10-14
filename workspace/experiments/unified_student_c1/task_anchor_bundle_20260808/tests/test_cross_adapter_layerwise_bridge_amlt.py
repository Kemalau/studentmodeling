from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from render_cross_adapter_layerwise_bridge_c4 import load, render  # noqa: E402
from validate_cross_adapter_layerwise_bridge_amlt import validate  # noqa: E402


class CrossAdapterLayerwiseBridgeAmltTest(unittest.TestCase):
    def test_four_matched_jobs_validate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        value = render(
            load(root / "amlt/feeds_one_shot_layer_transport_c4_20260731.yaml")
        )
        validate(value)
        self.assertEqual(len(value["jobs"]), 4)
        for job in value["jobs"]:
            self.assertIn("--decode-adapter", str(job["command"]))
            self.assertNotIn("train_recurrent_student", str(job["command"]))


if __name__ == "__main__":
    unittest.main()
