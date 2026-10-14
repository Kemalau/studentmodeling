from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Checkpoint750LocalFormalScreenTest(unittest.TestCase):
    def test_strict_state_route_and_frozen_support_are_preserved(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_checkpoint750_local_formal_screen.sh").read_text()
        self.assertIn("checkpoint_completion.json", text)
        self.assertIn("receipt.get('round')==750", text)
        self.assertIn('PYTHONPATH="scripts/frozen_layerwise_finalfit:scripts"', text)
        self.assertIn("--state-architecture layerwise", text)
        self.assertIn("--max-samples-per-group 25", text)
        self.assertIn("predict_correctness predict_error_persistence diagnose_misconception", text)
        self.assertIn("predict_response predict_prospective_error", text)
        self.assertNotIn("amlt run", text)
        self.assertNotIn("raw history", text.lower())


if __name__ == "__main__":
    unittest.main()
