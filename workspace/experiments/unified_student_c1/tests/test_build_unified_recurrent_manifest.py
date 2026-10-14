from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_unified_recurrent_manifest import (  # noqa: E402
    make_eedi_mining_record,
    make_history_record,
)


def base_row(source: str, history: list[dict], task: str = "predict_response") -> dict:
    return {
        "example_id": "x",
        "student_id": "s",
        "task": task,
        "metadata": {"mix_source": source, "question_id": "q"},
        "history": history,
        "prompt": (
            "instruction\n<student_history>\nTrial 1:\none\n\n"
            "Trial 2:\ntwo\n</student_history>\n<STUDENT_STATE>\n<query>future</query>"
        ),
        "query": {"query_text": "Question Q\nCorrect A\nIncorrect B"},
        "target": "<answer>A</answer>",
        "target_label": "A",
        "candidate_labels": ["A", "B"],
        "candidate_targets": {"A": "<answer>A</answer>", "B": "<answer>B</answer>"},
    }


class UnifiedManifestTest(unittest.TestCase):
    def test_temporal_history_caps_last_updates(self) -> None:
        history = [
            {"question_text": "old", "student_response": "x"},
            {"question_text": "new", "student_response": "y"},
        ]
        record = make_history_record(base_row("dbe_kt22", history), 1)
        self.assertEqual(record["observed_interactions"], 1)
        self.assertEqual(record["sequence_semantics"], "chronological_trajectory")
        self.assertNotIn("Trial 2", record["direct_text"])
        self.assertIn("two", record["direct_text"])
        self.assertIn("Trial 2", record["full_context_text"])
        self.assertIn("one", record["full_context_text"])

    def test_profile_uses_fixed_budget_without_temporal_claim(self) -> None:
        history = [
            {"question_text": "one", "student_response": "x"},
            {"question_text": "two", "student_response": "y"},
        ]
        record = make_history_record(base_row("eedi_response", history), 1)
        self.assertEqual(record["observed_interactions"], 1)
        self.assertEqual(len(record["update_texts"]), 1)
        self.assertIn("non_temporal", record["sequence_semantics"])

    def test_empty_history_is_a_cold_start_not_dropped(self) -> None:
        row = base_row("foundationalassist", [], "predict_error_persistence")
        row["prompt"] = (
            "instruction\n<student_history>\n</student_history>\n"
            "<STUDENT_STATE>\n<query>future</query>"
        )
        record = make_history_record(row, 4)
        self.assertEqual(record["sequence_semantics"], "cold_start_no_observation")
        self.assertEqual(record["observed_interactions"], 0)
        self.assertEqual(record["update_texts"], [])

    def test_eedi_mining_routes_wrong_response_through_state(self) -> None:
        row = base_row("eedi_mining", [], "diagnose_misconception")
        row["prompt"] = (
            "diagnose\n<student_history>\n</student_history>\n"
            "<STUDENT_STATE>\n<query>raw shortcut</query>"
        )
        record = make_eedi_mining_record(row)
        self.assertIn("Incorrect B", record["update_texts"][0])
        self.assertNotIn("Incorrect B", record["decode_text"])
        self.assertNotIn("raw shortcut", record["decode_text"])
        self.assertIn("Incorrect B", record["direct_text"])
        self.assertEqual(record["full_context_text"], record["direct_text"])


if __name__ == "__main__":
    unittest.main()
