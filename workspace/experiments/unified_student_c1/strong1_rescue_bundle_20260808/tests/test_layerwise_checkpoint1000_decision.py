from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_checkpoint1000_decision import summarize  # noqa: E402


def c4(passed: bool) -> dict:
    return {
        "stage": "layerwise_finalfit_checkpoint1000_c4",
        "diagnostic_only": True,
        "formal_final_c4_required": True,
        "exact_resume_ready": True,
        "scientific_gate_pass_at_checkpoint": passed,
        "clean_n": 80,
        "swap": 0.35,
        "random": 0.15,
        "same_rule": 0.05,
        "noop": 0.0,
        "D_specific": 0.20 if passed else 0.05,
    }


def formal(passed: bool) -> dict:
    return {
        "stage": "layerwise_finalfit_checkpoint1000_formal_screen",
        "diagnostic_only": True,
        "formal_final_evaluation_required": True,
        "diagnostic_gate_pass": passed,
        "sampled_rows": 487,
        "performance": {"predict_response": {"pass": passed}},
        "T3_format_valid_rate": 1.0,
    }


def t3_semantic(passed: bool) -> dict:
    return {
        "stage": "layerwise_finalfit_checkpoint1000_T3_semantic_judge",
        "diagnostic_only": True,
        "support": 75,
        "pass": passed,
        "semantic_accuracy": 0.5 if passed else 0.1,
        "retention_ratio": 1.2 if passed else 0.2,
        "minimum_retention_ratio": 0.85,
        "by_group": {},
    }


class Round1000DecisionTest(unittest.TestCase):
    def test_four_scientific_outcomes_are_explicit(self) -> None:
        expected = {
            (True, True): "healthy_early_checkpoint",
            (False, True): "causal_specificity_at_risk",
            (True, False): "task_retention_at_risk",
            (False, False): "joint_training_instability_at_risk",
        }
        for flags, diagnosis in expected.items():
            with self.subTest(flags=flags):
                value = summarize(c4(flags[0]), formal(flags[1]), t3_semantic(True))
                self.assertEqual(value["diagnosis"], diagnosis)
                self.assertTrue(value["main_fit_must_continue"])
                self.assertTrue(value["C4"]["exact_resume_ready"])
                self.assertEqual(value["rescue_preparation_required"], not all(flags))

    def test_checkpoint_without_exact_resume_proof_is_rejected(self) -> None:
        value = c4(True)
        value["exact_resume_ready"] = False
        with self.assertRaisesRegex(ValueError, "exact-resume"):
            summarize(value, formal(True), t3_semantic(True))

    def test_semantic_failure_overrides_format_only_formal_pass(self) -> None:
        value = summarize(c4(True), formal(True), t3_semantic(False))
        self.assertEqual(value["diagnosis"], "task_retention_at_risk")
        self.assertFalse(value["formal_screen"]["pass"])
        self.assertFalse(value["T3_semantic"]["pass"])
        self.assertTrue(value["rescue_preparation_required"])

    def test_watcher_never_mutates_or_submits_main_fit(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_checkpoint1000_decision.sh").read_text()
        self.assertIn("rescue_preparation_required", text)
        self.assertIn("main_fit_must_continue", text)
        self.assertIn('[[ -f "$auth_required" ]]', text)
        self.assertIn("attention_is_current", text)
        self.assertIn('"$active_pid" -nt "$marker"', text)
        self.assertIn('active_experiment.txt" -nt "$marker"', text)
        self.assertNotIn("amlt run", text)
        self.assertNotIn("amlt cancel", text)
        self.assertIn("checkpoint1000_t3_semantic_judge", text)


if __name__ == "__main__":
    unittest.main()
