from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_promoted_round1250_same_rule_repair_manifest import (  # noqa: E402
    SEGMENT_ROWS,
    START_GLOBAL_POSITION,
    TASK_COUNTS,
)


class BuildPromotedRound1250SameRuleRepairManifestTest(unittest.TestCase):
    def test_frozen_scope(self) -> None:
        self.assertEqual(START_GLOBAL_POSITION, 20_000)
        self.assertEqual(SEGMENT_ROWS, 1_000)
        self.assertEqual(sum(TASK_COUNTS.values()), 1_000)
        self.assertEqual(TASK_COUNTS["predict_error_persistence"], 600)
        self.assertEqual(TASK_COUNTS["diagnose_misconception"], 200)


if __name__ == "__main__":
    unittest.main()
