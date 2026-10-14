import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_t3_semantic_result import EXPECTED_SUPPORT, validate


class T3SemanticResultTest(unittest.TestCase):
    def fixture(self) -> dict:
        return {
            "generation_rows": 1494,
            "judged_rows": 1494,
            "judge_errors": 0,
            "judge_backend": "trapi",
            "judge_effective_model": "gpt-4o_2024-11-20",
            "judge_rubric": "t3-semantic-equivalence-v2-substantive-schema",
            "judge_rubric_sha256": "a" * 64,
            "metrics_by_group": {
                name: {
                    "n": support,
                    "llm_judge_n": support,
                    "llm_judge_api_or_cache": support,
                    "llm_judge_exact_shortcuts": 0,
                    "llm_judge_accuracy": 0.5,
                    "llm_judge_raw_semantic_accuracy": 0.5,
                    "llm_judge_schema_valid_n": support,
                    "llm_judge_mean_confidence": 0.9,
                    "natural_schema_valid_rate": 1.0,
                }
                for name, support in EXPECTED_SUPPORT.items()
            },
        }

    def write(self, root: Path, value: dict) -> Path:
        path = root / "metrics.json"
        path.write_text(json.dumps(value))
        return path

    def test_complete_fixed_panel_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = validate(self.write(Path(directory), self.fixture()))
            self.assertTrue(result["pass"])
            self.assertEqual(result["support"], 1494)
            self.assertTrue(result["group_floor_pass"])
            self.assertEqual(
                result["weighting_scheme"], "formal_panel_group_prevalence_v1"
            )

    def test_strong_synthetic_groups_cannot_hide_real_eedi_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            value["metrics_by_group"][
                "diagnose_misconception/eedi_mining/open_taxonomy_generation"
            ]["llm_judge_accuracy"] = 0.01
            for name in (
                "diagnose_misconception/malrule/synthetic_cross_template_open_generation",
                "diagnose_misconception/malrule/synthetic_same_template_open_generation",
            ):
                value["metrics_by_group"][name]["llm_judge_accuracy"] = 1.0
            result = validate(self.write(Path(directory), value))
            self.assertGreater(result["retention_ratio"], 0.85)
            self.assertFalse(result["group_floor_pass"])
            self.assertFalse(result["pass"])

    def test_missing_judgments_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            value["judged_rows"] -= 1
            with self.assertRaisesRegex(ValueError, "Not every"):
                validate(self.write(Path(directory), value))

    def test_non_trapi_backend_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            value["judge_backend"] = "responses"
            with self.assertRaisesRegex(ValueError, "requires TRAPI"):
                validate(self.write(Path(directory), value))

    def test_exact_match_shortcut_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            group = next(iter(value["metrics_by_group"].values()))
            group["llm_judge_api_or_cache"] -= 1
            group["llm_judge_exact_shortcuts"] = 1
            with self.assertRaisesRegex(ValueError, "independent semantic judgments"):
                validate(self.write(Path(directory), value))

    def test_semantic_collapse_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            for group in value['metrics_by_group'].values():
                group['llm_judge_accuracy'] = 0.1
            self.assertFalse(validate(self.write(Path(directory), value))['pass'])

    def test_legacy_only_schema_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = self.fixture()
            group = value["metrics_by_group"][
                "diagnose_misconception/malrule/synthetic_same_template_open_generation"
            ]
            group["natural_schema_valid_rate"] = 0.0
            result = validate(self.write(Path(directory), value))
            self.assertFalse(result["natural_schema_pass"])
            self.assertFalse(result["pass"])

    def test_cli_returns_nonzero_on_semantic_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = self.fixture()
            for group in value['metrics_by_group'].values():
                group['llm_judge_accuracy'] = 0.1
            source = self.write(root, value)
            result = subprocess.run(
                [sys.executable, str(ROOT / 'scripts/validate_t3_semantic_result.py'), '--metrics', str(source), '--output', str(root / 'gate.json')],
                check=False,
            )
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
