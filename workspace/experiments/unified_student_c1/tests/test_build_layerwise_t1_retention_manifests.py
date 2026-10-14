from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_layerwise_t1_retention_manifests import BRANCH_SPECS, SEGMENT_ROWS  # noqa: E402


class T1RetentionManifestTest(unittest.TestCase):
    def test_each_branch_has_exact_support(self) -> None:
        for branch, spec in BRANCH_SPECS.items():
            with self.subTest(branch=branch):
                self.assertEqual(
                    sum(sum(sources.values()) for sources in spec.values()),
                    SEGMENT_ROWS,
                )

    def test_t1_heavy_and_balanced_all_are_distinct(self) -> None:
        heavy = BRANCH_SPECS["t1_heavy"]
        balanced = BRANCH_SPECS["balanced_all"]
        self.assertEqual(sum(heavy["predict_correctness"].values()), 650)
        self.assertEqual(sum(balanced["predict_correctness"].values()), 400)
        self.assertEqual(sum(balanced["diagnose_misconception"].values()), 250)
        self.assertEqual(sum(balanced["predict_error_persistence"].values()), 250)


if __name__ == "__main__":
    unittest.main()
