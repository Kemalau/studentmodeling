from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_unified_final_eval import FROZEN_GROUP_SUPPORT, summarize


METRIC_BY_TASK = {
    "predict_correctness": {"macro_f1": 0.7},
    "predict_error_persistence": {"macro_f1": 0.6},
    "predict_response": {"math_equivalence": 0.55},
    "diagnose_misconception": {"format_valid_rate": 1.0},
    "predict_prospective_error": {"joint_math_equivalence": 0.8},
}


def write_fixture(root: Path) -> None:
    task_sets = (
        {"predict_correctness", "predict_error_persistence"},
        {"predict_response"},
        {"diagnose_misconception"},
        {"predict_prospective_error"},
    )
    for index, tasks in enumerate(task_sets):
        groups = {}
        for name, support in FROZEN_GROUP_SUPPORT.items():
            task = name.split("/", 1)[0]
            if task in tasks:
                groups[name] = {"n": support, **METRIC_BY_TASK[task]}
        directory = root / f"j{index}"
        directory.mkdir()
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "state_architecture": "layerwise",
                    "state_shape": [64, 5120],
                    "raw_history_visible_to_decoder": False,
                    "persistent_history_kv_cache": False,
                    "sampled_rows": sum(row["n"] for row in groups.values()),
                    "metrics_by_group": groups,
                }
            )
        )
    for seed in (13, 42, 97):
        directory = root / f"c4_seed{seed}"
        directory.mkdir()
        (directory / "recurrent_causal_gate.json").write_text(
            json.dumps({
                "state_architecture": "layerwise",
                "state_shape": [64, 5120],
                "raw_history_visible_to_decoder": False,
                "persistent_history_kv_cache": False,
                "pair_manifest": f"manifests/c4_stage_a_gate_pairs_seed{seed}_n150_v1.json",
                "summary": {
                    "all_pairs_n": 150,
                    "clean_n": 80,
                    "D_specific": 0.2,
                    "scientific_gate_pass": True,
                    "checks": {"candidate_support": True, "specificity": True, "noop_control": True, "same_rule_control": True},
                    "clean": {
                        "swap": {"flip_to_A_rate": 0.3},
                        "random": {"flip_to_A_rate": 0.1},
                        "samemal": {"flip_to_A_rate": 0.05},
                        "noop": {"flip_to_A_rate": 0.0},
                    },
                },
            })
        )


class SummaryTest(unittest.TestCase):
    def test_positive_summary_requires_every_frozen_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            value = summarize(root)
            self.assertTrue(value["scientific_gate_pass"])
            self.assertTrue(value["checks"]["full_formal_row_support"])
            self.assertTrue(value["checks"]["all_protocol_support_exact"])
            self.assertTrue(value["checks"]["C4_replication"])
            self.assertEqual(value["C4"]["passing_seeds"], 3)

    def test_missing_pair_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            next((root / "c4_seed13").iterdir()).unlink()
            with self.assertRaisesRegex(ValueError, "three C4"):
                summarize(root)

    def test_one_failed_pair_seed_still_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = root / "c4_seed13/recurrent_causal_gate.json"
            value = json.loads(path.read_text())
            value["summary"]["scientific_gate_pass"] = False
            value["summary"]["clean"]["samemal"]["flip_to_A_rate"] = 0.11
            value["summary"]["checks"]["same_rule_control"] = False
            path.write_text(json.dumps(value))
            summary = summarize(root)
            self.assertTrue(summary["scientific_gate_pass"])
            self.assertEqual(summary["C4"]["passing_seeds"], 2)

    def test_two_failed_pair_seeds_fail_replication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            for seed in (13, 97):
                path = root / f"c4_seed{seed}/recurrent_causal_gate.json"
                value = json.loads(path.read_text())
                value["summary"]["scientific_gate_pass"] = False
                value["summary"]["clean"]["samemal"]["flip_to_A_rate"] = 0.11
                value["summary"]["checks"]["same_rule_control"] = False
                path.write_text(json.dumps(value))
            summary = summarize(root)
            self.assertFalse(summary["scientific_gate_pass"])
            self.assertFalse(summary["checks"]["C4_replication"])

    def test_missing_protocol_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = next(root.rglob("metrics.json"))
            value = json.loads(path.read_text())
            name = next(iter(value["metrics_by_group"]))
            value["metrics_by_group"][name]["n"] -= 1
            value["sampled_rows"] -= 1
            path.write_text(json.dumps(value))
            summary = summarize(root)
            self.assertFalse(summary["scientific_gate_pass"])
            self.assertFalse(summary["checks"]["all_protocol_support_exact"])
            self.assertFalse(summary["checks"]["full_formal_row_support"])

    def test_task_collapse_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_fixture(root)
            path = next(root.rglob("metrics.json"))
            value = json.loads(path.read_text())
            value["metrics_by_group"] = {}
            value["sampled_rows"] = 0
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Task groups drift"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
