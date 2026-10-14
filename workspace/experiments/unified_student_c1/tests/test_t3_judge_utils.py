from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_formal_protocols import summarize as heavyweight_summarize  # noqa: E402
from t3_judge_utils import natural_schema_valid, redacted_records, summarize_t3  # noqa: E402
from t3_llm_judge import JudgeConfig, judge_t3_records  # noqa: E402


class T3JudgeUtilsTest(unittest.TestCase):
    def records(self) -> list[dict]:
        return [
            {
                "example_id": "one",
                "task": "diagnose_misconception",
                "source": "malrule",
                "protocol": "synthetic_same_template_open_generation",
                "raw_output": "<misconception>A &amp; B</misconception>",
                "target": "<misconception>a & b</misconception>",
                "llm_judge_correct": True,
                "llm_judge_confidence": 1.0,
                "llm_judge_mode": "exact_match_shortcut",
            },
            {
                "example_id": "two",
                "task": "diagnose_misconception",
                "source": "malrule",
                "protocol": "synthetic_same_template_open_generation",
                "raw_output": "<misconception>wrong</misconception>",
                "target": "<misconception>right</misconception>",
                "llm_judge_correct": False,
                "llm_judge_confidence": 0.9,
                "llm_judge_mode": "api",
            },
        ]

    def test_torch_free_summary_matches_formal_evaluator(self) -> None:
        records = self.records()
        self.assertEqual(summarize_t3(records), heavyweight_summarize(records))

    def test_redaction_matches_required_hash_contract(self) -> None:
        row = redacted_records(self.records())[0]
        self.assertNotIn("raw_output", row)
        self.assertNotIn("target", row)
        self.assertEqual(len(row["raw_output_sha256"]), 64)
        self.assertEqual(len(row["target_sha256"]), 64)

    def test_judge_entrypoint_imports_in_torchless_fkft_environment(self) -> None:
        python = Path("/home/v-yuejiali/fkft/.venv/bin/python")
        if not python.is_file():
            self.skipTest("fkft environment unavailable")
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import sys;"
                    f"sys.path.insert(0,{str(ROOT / 'scripts')!r});"
                    "import judge_formal_t3_predictions;"
                    "assert 'torch' not in sys.modules"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_semantic_judge_does_not_shortcut_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                {
                    "example_id": "example-1",
                    "task": "diagnose_misconception",
                    "raw_output": "<diagnosis>same rule</diagnosis>",
                    "target": "<diagnosis>same rule</diagnosis>",
                }
            ]
            rows = {"example-1": {"direct_text": "student evidence"}}
            result = {
                "equivalent": True,
                "confidence": 0.99,
                "rationale": "Semantically equivalent.",
                "judge_response_id": "mock",
                "usage": {},
            }
            config = JudgeConfig(
                model="mock-judge",
                base_url="http://invalid",
                api_key="unused",
                workers=1,
            )
            with patch("t3_llm_judge._call_with_retries", return_value=result) as call:
                judge_t3_records(
                    records,
                    rows,
                    config,
                    root / "cache.jsonl",
                    lambda value: value.split("<diagnosis>", 1)[1].split("</diagnosis>", 1)[0],
                    lambda value: str(value).strip().casefold(),
                )
            call.assert_called_once()
            self.assertEqual(records[0]["llm_judge_mode"], "api")

    def test_natural_schema_is_stricter_than_legacy_diagnosis_format(self) -> None:
        self.assertTrue(
            natural_schema_valid(
                "<label>fractions.rule</label>"
                "<diagnosis>Adds the denominators.</diagnosis>"
                "<evidence>Answered 2/6 instead of 1/2.</evidence>"
            )
        )
        self.assertFalse(natural_schema_valid("<misconception>y</misconception>"))
        self.assertFalse(
            natural_schema_valid("<label>x</label><diagnosis>y</diagnosis>")
        )
        self.assertFalse(
            natural_schema_valid(
                "<label>none.none</label>"
                "<diagnosis>natural-language diagnosis</diagnosis>"
                "<evidence>interaction-grounded evidence</evidence>"
            )
        )
        self.assertFalse(
            natural_schema_valid(
                "<misconception_label_contract>instructions</misconception_label_contract>"
                "<label>fractions.rule</label>"
                "<diagnosis>Adds the denominators.</diagnosis>"
                "<evidence>Answered 2/6 instead of 1/2.</evidence>"
            )
        )

    def test_semantic_equivalence_is_gated_by_substantive_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            records = [
                {
                    "example_id": "placeholder",
                    "task": "diagnose_misconception",
                    "raw_output": (
                        "<label>none.none</label>"
                        "<diagnosis>natural-language diagnosis</diagnosis>"
                        "<evidence>interaction-grounded evidence</evidence>"
                    ),
                    "target": (
                        "<label>fractions.denominator_comparison_error</label>"
                        "<diagnosis>Compares fractions only by denominator size.</diagnosis>"
                        "<evidence>Ranks 1/10 above 1/5.</evidence>"
                    ),
                }
            ]
            config = JudgeConfig(
                model="mock-judge", base_url="http://invalid", api_key="unused", workers=1
            )
            api_result = {
                "equivalent": True,
                "confidence": 0.99,
                "rationale": "Equivalent.",
                "judge_response_id": "mock",
                "usage": {},
            }
            with patch("t3_llm_judge._call_with_retries", return_value=api_result):
                judge_t3_records(
                    records,
                    {"placeholder": {"direct_text": "student evidence"}},
                    config,
                    Path(temporary) / "cache.jsonl",
                    lambda value: value.split("<diagnosis>", 1)[1].split("</diagnosis>", 1)[0],
                    lambda value: str(value).strip().casefold(),
                )
            self.assertTrue(records[0]["llm_judge_semantic_equivalent"])
            self.assertFalse(records[0]["llm_judge_schema_valid"])
            self.assertFalse(records[0]["llm_judge_correct"])


if __name__ == "__main__":
    unittest.main()
