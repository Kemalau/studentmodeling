from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from split_full_mix_by_original_split import split_full_mix  # noqa: E402


class SplitFullMixTest(unittest.TestCase):
    def test_preserves_prompt_complete_frozen_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "all.jsonl"
            rows = [
                {
                    "example_id": f"e{index}",
                    "student_id": f"{split}-student",
                    "task": "predict_correctness",
                    "dataset": "demo",
                    "prompt": "<STUDENT_STATE> prompt",
                    "split": "train",
                    "metadata": {"original_split": split, "mix_source": "demo"},
                }
                for index, split in enumerate(("train", "valid"), start=1)
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = split_full_mix(source, root / "split")
            self.assertEqual(result["total_rows"], 2)
            self.assertEqual(result["train_valid_student_overlap"], {"demo": 0})
            valid = json.loads((root / "split/valid.jsonl").read_text())
            self.assertEqual(valid["split"], "valid")
            self.assertIn("prompt", valid)

    def test_rejects_student_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "all.jsonl"
            rows = [
                {
                    "example_id": f"e{index}",
                    "student_id": "same",
                    "task": "predict_correctness",
                    "dataset": "demo",
                    "prompt": "<STUDENT_STATE> prompt",
                    "metadata": {"original_split": split, "mix_source": "demo"},
                }
                for index, split in enumerate(("train", "valid"), start=1)
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "student overlap"):
                split_full_mix(source, root / "split")


if __name__ == "__main__":
    unittest.main()
