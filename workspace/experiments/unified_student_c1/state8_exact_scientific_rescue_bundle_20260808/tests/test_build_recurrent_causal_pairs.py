from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_recurrent_causal_pairs import build_split  # noqa: E402


def source_row(example_id: str, answer: str, other: str) -> dict:
    history = [
        {
            "interaction_id": f"{example_id}-h",
            "problem_text": "Observed problem",
            "student_answer": answer,
            "correct_answer": other,
            "is_correct": False,
        }
    ]
    instruction = "Predict response.\n"
    history_text = (
        "<student_history>\nTrial 1:\nObserved problem\n</student_history>\n"
    )
    question = (
        "<query>Next problem\nValid outputs:\n"
        f"<predicted_response>{answer}</predicted_response>\n"
        f"<predicted_response>{other}</predicted_response>\n</query>\nAnswer:"
    )
    return {
        "example_id": example_id,
        "history": history,
        "seg_instruction": instruction,
        "seg_history": history_text,
        "seg_question": question,
        "prompt": instruction + history_text + "<STUDENT_STATE>\n" + question,
        "target_label": answer,
        "candidate_labels": [answer, other],
        "candidate_targets": {
            answer: f"<predicted_response>{answer}</predicted_response>",
            other: f"<predicted_response>{other}</predicted_response>",
        },
    }


class RecurrentCausalPairBuilderTest(unittest.TestCase):
    def test_builds_closed_native_and_donor_routes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [source_row("donor", "A", "B"), source_row("recipient", "B", "A")]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "task": "malrule_next_response",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "pairs": [
                            {
                                "donor_id": "donor",
                                "recipient_id": "recipient",
                                "donor_label": "rule_a",
                                "recipient_label": "rule_b",
                                "donor_target_on_recipient_query": "A",
                                "recipient_target": "B",
                                "family": "toy",
                            }
                        ],
                    }
                )
            )
            output = root / "out.jsonl"
            stats = build_split(source, manifest, output, max_updates=1)
            record = json.loads(output.read_text())
            self.assertEqual(stats["pairs"], 1)
            self.assertEqual(record["donor_target_label"], "A")
            self.assertEqual(record["native_target_label"], "B")
            self.assertEqual(record["decode_text"].count("<STUDENT_STATE>"), 1)
            self.assertNotIn("<student_history>", record["decode_text"].lower())
            self.assertEqual(len(record["donor_update_texts"]), 1)
            self.assertEqual(record["donor_update_texts"][0].count("<STUDENT_STATE>"), 2)

    def test_rejects_no_answer_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [source_row("donor", "A", "B"), source_row("recipient", "B", "A")]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest = root / "pairs.json"
            manifest.write_text(
                json.dumps(
                    {
                        "task": "malrule_next_response",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "pairs": [
                            {
                                "donor_id": "donor",
                                "recipient_id": "recipient",
                                "donor_label": "rule_a",
                                "recipient_label": "rule_b",
                                "donor_target_on_recipient_query": "B",
                                "recipient_target": "B",
                                "family": "toy",
                            }
                        ],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "no answer intervention"):
                build_split(source, manifest, root / "out.jsonl", max_updates=1)


if __name__ == "__main__":
    unittest.main()
