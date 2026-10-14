from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m2_seeded_cf4_screen import (  # noqa: E402
    EVALUATOR_SHA256,
    M2_RUN_DIR,
    TRAINER_SHA256,
    load,
    render,
)
from validate_m2_seeded_cf4_screen_amlt import validate  # noqa: E402


class M2SeededCF4Test(unittest.TestCase):
    def test_only_initialization_differs_from_matched_screen(self):
        value = render(load(ROOT / "amlt/feeds_m4_cf4_single_hedge_20260731.yaml"))
        validate(value)
        command = value["jobs"][0]["command"]
        self.assertIn(M2_RUN_DIR, command)
        self.assertNotIn("M3_RUN_DIR", command)
        self.assertIn("--diagnostic-max-micro-steps 1000", command)
        self.assertIn("--causal-objective-mode donor_directional", command)
        self.assertIn(TRAINER_SHA256, command)
        self.assertIn(EVALUATOR_SHA256, command)


if __name__ == "__main__":
    unittest.main()
