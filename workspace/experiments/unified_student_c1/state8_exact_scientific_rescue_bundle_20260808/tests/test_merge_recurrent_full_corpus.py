from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from merge_recurrent_full_corpus import merge  # noqa: E402


def row(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "task": "predict_correctness",
        "source": "demo",
        "decode_text": "<STUDENT_STATE>\n<query>x</query>",
    }


class MergeRecurrentFullCorpusTest(unittest.TestCase):
    def test_merges_unique_rows_and_marks_source_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, valid = root / "train.jsonl", root / "valid.jsonl"
            train.write_text(json.dumps(row("a")) + "\n")
            valid.write_text(json.dumps(row("b")) + "\n")
            result = merge(train, valid, root / "all.jsonl", root / "manifest.json")
            self.assertEqual(result["unique_examples"], 2)
            merged = [json.loads(line) for line in (root / "all.jsonl").read_text().splitlines()]
            self.assertEqual([item["final_fit_source_split"] for item in merged], ["train", "valid"])

    def test_rejects_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, valid = root / "train.jsonl", root / "valid.jsonl"
            train.write_text(json.dumps(row("same")) + "\n")
            valid.write_text(json.dumps(row("same")) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                merge(train, valid, root / "all.jsonl", root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
