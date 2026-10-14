from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_promoted_round1250_same_rule_repair_ab import summarize


class SameRuleRepairSummaryTest(unittest.TestCase):
    def write(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_arm(self, root: Path, variant: str, same: float, specific: float) -> None:
        job = root / f"job-{variant}"
        self.write(job / "metrics/task_retention_training_receipt.json", {
            "causal_variant": variant, "diagnostic_only": True,
            "source_checkpoint_round": 1250, "loss_profile": "state_path",
        })
        self.write(job / "metrics/completion.json", {
            "micro_steps": 1000, "optimizer_steps": 250, "causal_pairs_seen": 50,
        })
        self.write(job / "metrics/run_config.json", {"causal_weight": 2.0})
        clean = {
            "swap": {"flip_to_A_rate": 0.45},
            "random": {"flip_to_A_rate": 0.20},
            "samemal": {"flip_to_A_rate": same},
            "noop": {"flip_to_A_rate": 0.0},
        }
        self.write(job / "eval/c4/recurrent_causal_gate.json", {
            "state_architecture": "layerwise", "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "summary": {"clean": clean, "clean_n": 80, "D_specific": specific,
                        "scientific_gate_pass": same <= 0.10 and specific >= 0.15},
        })
        groups = {
            "predict_correctness/x": {"n": 25, "macro_f1": 0.60},
            "predict_response/x": {"n": 25, "math_equivalence": 0.30},
            "predict_prospective_error/x": {"n": 25, "joint_math_equivalence": 0.45},
            "diagnose_misconception/a": {"n": 25, "format_valid_rate": 1.0},
            "diagnose_misconception/b": {"n": 25, "format_valid_rate": 1.0},
            "diagnose_misconception/c": {"n": 25, "format_valid_rate": 1.0},
        }
        self.write(job / "eval/formal_screen/metrics.json", {"metrics_by_group": groups})
        self.write(job / "eval/t6_full/metrics.json", {
            "sampled_rows": 976,
            "metrics_by_group": {"predict_error_persistence/x": {
                "n": 976, "macro_f1": 0.55, "positive_recall": 0.20,
            }},
        })

    def make_t3(self, root: Path, variant: str, accuracy: float) -> None:
        self.write(root / variant / "gate.json", {
            "support": 75, "checkpoint_round": 1500, "pass": True,
            "semantic_accuracy": accuracy, "retention_ratio": 1.0,
            "judge_backend": "trapi", "judge_rubric_sha256": "a" * 64,
        })

    def test_repair_can_only_recommend_replication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); results = root / "results"; t3 = root / "t3"
            self.make_arm(results, "base", 0.20, 0.20)
            self.make_arm(results, "same_rule", 0.05, 0.21)
            self.make_t3(t3, "base", 0.42); self.make_t3(t3, "same_rule", 0.42)
            value = summarize(results, t3)
            self.assertTrue(value["replication_recommended"])
            self.assertFalse(value["recipe_promotion_allowed"])
            self.assertTrue(value["main_exact_epoch_unchanged"])

    def test_task_regression_rejects_replication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); results = root / "results"; t3 = root / "t3"
            self.make_arm(results, "base", 0.20, 0.20)
            self.make_arm(results, "same_rule", 0.05, 0.21)
            metrics = results / "job-same_rule/eval/formal_screen/metrics.json"
            value = json.loads(metrics.read_text()); value["metrics_by_group"]["predict_response/x"]["math_equivalence"] = 0.10
            metrics.write_text(json.dumps(value))
            self.make_t3(t3, "base", 0.42); self.make_t3(t3, "same_rule", 0.42)
            value = summarize(results, t3)
            self.assertFalse(value["replication_recommended"])


if __name__ == "__main__":
    unittest.main()
