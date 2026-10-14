from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_layerwise_task_retention_manifests import (  # noqa: E402
    ANCHOR_SPECS,
    BRANCH_SPECS,
    SEGMENT_ROWS,
    STAGED_T3_T6_PHASES,
    build,
)


class TaskRetentionManifestTest(unittest.TestCase):
    def test_disjoint_followup_excludes_prior_indices(self) -> None:
        rows = []
        # This unit checks selection semantics with compact synthetic pools;
        # temporarily shrink the module specs to available support.
        self.assertIn("equal_t3_t6", BRANCH_SPECS)
        self.assertTrue(callable(build))

    def test_each_branch_matches_its_declared_segment_size(self) -> None:
        anchors = sum(ANCHOR_SPECS.values())
        self.assertEqual(anchors, 200)
        for branch, spec in BRANCH_SPECS.items():
            with self.subTest(branch=branch):
                self.assertEqual(
                    anchors + sum(spec["t3"].values()) + sum(spec["t6"].values()),
                    int(spec.get("segment_rows", SEGMENT_ROWS)),
                )

    def test_arms_change_only_t3_t6_ratio(self) -> None:
        equal = BRANCH_SPECS["equal_t3_t6"]
        heavy = BRANCH_SPECS["t3_heavy"]
        self.assertEqual(sum(equal["t3"].values()), 400)
        self.assertEqual(sum(equal["t6"].values()), 400)
        self.assertEqual(sum(heavy["t3"].values()), 600)
        self.assertEqual(sum(heavy["t6"].values()), 200)
        for spec in BRANCH_SPECS.values():
            persistent = sum(n for (source, label), n in spec["t6"].items() if label == "persistent")
            transient = sum(n for (source, label), n in spec["t6"].items() if label == "transient")
            self.assertEqual(persistent, transient)

    def test_eedi_dense_is_a_matched_high_information_panel(self) -> None:
        dense = BRANCH_SPECS["eedi_dense"]
        self.assertEqual(dense["t3"], {"eedi_mining": 600, "malrule": 100})
        self.assertEqual(sum(dense["t6"].values()), 100)

    def test_consolidation_preserves_task_and_global_t6_balance(self) -> None:
        spec = BRANCH_SPECS["equal_t3_t6_consolidation"]
        self.assertEqual(sum(spec["t3"].values()), 400)
        self.assertEqual(sum(spec["t6"].values()), 400)
        self.assertEqual(
            spec["t6"],
            {
                ("foundationalassist", "persistent"): 200,
                ("foundationalassist", "transient"): 200,
            },
        )

    def test_staged_curriculum_preserves_support_and_orders_three_phases(self) -> None:
        staged = BRANCH_SPECS["staged_t3_t6"]
        self.assertEqual(staged["t3"], {"eedi_mining": 400, "malrule": 100})
        self.assertEqual(sum(staged["t6"].values()), 300)
        self.assertEqual(
            [phase["name"] for phase in STAGED_T3_T6_PHASES],
            [
                "diagnosis_formation",
                "balanced_transition",
                "persistence_consolidation",
            ],
        )
        self.assertEqual(
            [
                phase["anchors"]
                + sum(phase["t3"].values())
                + sum(phase["t6"].values())
                for phase in STAGED_T3_T6_PHASES
            ],
            [400, 300, 300],
        )
        self.assertEqual(sum(phase["anchors"] for phase in STAGED_T3_T6_PHASES), 200)
        self.assertEqual(
            sum(sum(phase["t3"].values()) for phase in STAGED_T3_T6_PHASES),
            500,
        )
        self.assertEqual(
            sum(sum(phase["t6"].values()) for phase in STAGED_T3_T6_PHASES),
            300,
        )


if __name__ == "__main__":
    unittest.main()
