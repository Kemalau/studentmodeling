from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from render_m2_seeded_specificity_triplet_screen import (  # noqa: E402
    BRANCH,
    EVALUATOR_SHA256,
    JOB,
    RUN_DIR,
    TRAINER_SHA256,
    render,
)
from validate_m2_seeded_specificity_triplet_amlt import validate  # noqa: E402


class M2SeededSpecificityTripletScreenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = yaml.safe_load(
            (ROOT / "amlt/feeds_m2_seeded_cf4_screen_20260731.yaml").read_text()
        )

    def test_rendered_screen_is_single_gpu_and_c4_aligned(self) -> None:
        value = render(self.base)
        validate(value)
        self.assertEqual(len(value["jobs"]), 1)
        job = value["jobs"][0]
        self.assertEqual(job["name"], JOB)
        self.assertEqual(job["sku"], "80G1-A100")
        command = str(job["command"])
        for expected in (
            RUN_DIR,
            BRANCH,
            TRAINER_SHA256,
            EVALUATOR_SHA256,
            "--causal-objective-mode specificity_triplet",
            "initialization=m2_cf1_causal_adapter_and_state",
        ):
            self.assertIn(expected, command)
        self.assertNotIn("--causal-objective-mode donor_directional", command)

    def test_validator_rejects_old_trainer_hash(self) -> None:
        value = render(self.base)
        bad = copy.deepcopy(value)
        bad["jobs"][0]["command"] = str(bad["jobs"][0]["command"]).replace(
            TRAINER_SHA256, "0" * 64
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(bad)


if __name__ == "__main__":
    unittest.main()
