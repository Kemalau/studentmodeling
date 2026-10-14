from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_c3_exposure_matched_manifests import ARMS, build


class C3ManifestTest(unittest.TestCase):
    def test_exactly_one_factor_changes(self) -> None:
        train = []
        for index in range(8):
            train.append({
                "example_id": f"eedi-{index}", "task": "diagnose_misconception",
                "source": "eedi_mining", "original_metadata": {},
            })
        for index in range(8):
            for task in sorted(set(ARMS.values())):
                train.append({
                    "example_id": f"mal-{index}-{task}", "task": task,
                    "source": "malrule",
                    "original_metadata": {"raw_example_id": f"raw-{index}"},
                })
        values = build(
            train, [{"example_id": "formal-only"}], seed=7,
            start_global_position=24, eedi_count=4, auxiliary_count=4,
        )
        self.assertEqual(set(values), set(ARMS))
        shared = {
            (row["eedi_example_ids_sha256"], row["paired_malrule_raw_ids_sha256"], row["role_order_sha256"])
            for row in values.values()
        }
        self.assertEqual(len(shared), 1)
        for arm, row in values.items():
            self.assertEqual(row["segment_rows"], 8)
            self.assertEqual(row["formal_example_id_overlap"], 0)
            self.assertEqual(row["task_counts"]["diagnose_misconception"], 4 + (4 if arm == "malrule_misconception" else 0))
            self.assertEqual(row["source_counts"], {"eedi_mining": 4, "malrule": 4})

    def test_missing_paired_task_fails_closed(self) -> None:
        train = [{
            "example_id": "eedi", "task": "diagnose_misconception",
            "source": "eedi_mining", "original_metadata": {},
        }]
        with self.assertRaisesRegex(ValueError, "paired Malrule"):
            build(
                train, [], seed=1, start_global_position=4,
                eedi_count=1, auxiliary_count=1,
            )


if __name__ == "__main__":
    unittest.main()
