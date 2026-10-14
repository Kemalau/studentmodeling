from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from t3_resumable_progress import (  # noqa: E402
    atomic_write_jsonl,
    load_progress,
    progress_manifest,
)


class T3ResumableProgressTest(unittest.TestCase):
    def row(self, target: str = "gold") -> dict:
        return {
            "example_id": "row-1",
            "task": "diagnose_misconception",
            "source": "malrule",
            "protocol": "open",
            "delta": None,
            "target": target,
        }

    def test_progress_is_bound_to_key_rows_and_targets(self) -> None:
        rows = [self.row()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(load_progress(root, rows, "key-a"), {})
            record = {**rows[0], "raw_output": "diagnosis"}
            atomic_write_jsonl(root / "records.jsonl", [record])
            self.assertEqual(load_progress(root, rows, "key-a"), {"row-1": record})
            with self.assertRaisesRegex(ValueError, "manifest drift"):
                load_progress(root, rows, "key-b")

    def test_manifest_changes_with_target(self) -> None:
        first = progress_manifest([self.row("a")], "key")
        second = progress_manifest([self.row("b")], "key")
        self.assertNotEqual(
            first["row_signatures_sha256"], second["row_signatures_sha256"]
        )


if __name__ == "__main__":
    unittest.main()
