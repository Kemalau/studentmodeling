from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from audit_unified_recurrent_manifest import audit_manifests  # noqa: E402


TASKS = (
    "predict_correctness",
    "predict_response",
    "diagnose_misconception",
    "predict_prospective_error",
    "predict_error_persistence",
)


def record(index: int, task: str, split: str, student: str | None = None) -> dict:
    return {
        "example_id": f"{split}-{index}",
        "student_id": student or f"{split}-student-{index}",
        "task": task,
        "source": "eedi_mining" if student else "temporal",
        "sequence_semantics": "single_observed_wrong_response",
        "update_texts": [
            "<student_state_update><STUDENT_STATE>obs<STUDENT_STATE></student_state_update>"
        ],
        "decode_text": "instruction <STUDENT_STATE> query",
        "direct_text": "<student_history>obs</student_history> <STUDENT_STATE> query",
        "target": "answer",
    }


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class UnifiedRecurrentAuditTest(unittest.TestCase):
    def test_passes_and_ignores_only_non_student_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [record(i, task, "train") for i, task in enumerate(TASKS)]
            valid = [record(i, task, "valid") for i, task in enumerate(TASKS)]
            train[2] = record(2, TASKS[2], "train", "eedi_mining::NO_STUDENT_HISTORY")
            valid[2] = record(2, TASKS[2], "valid", "eedi_mining::NO_STUDENT_HISTORY")
            write(root / "train.jsonl", train)
            write(root / "valid.jsonl", valid)
            result = audit_manifests(root / "train.jsonl", root / "valid.jsonl")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["ignored_non_student_sentinels"]["eedi_mining"],
                ["eedi_mining::NO_STUDENT_HISTORY"],
            )

    def test_rejects_decoder_history_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [record(i, task, "train") for i, task in enumerate(TASKS)]
            valid = [record(i, task, "valid") for i, task in enumerate(TASKS)]
            train[0]["decode_text"] = "<student_history>leak</student_history><STUDENT_STATE>"
            write(root / "train.jsonl", train)
            write(root / "valid.jsonl", valid)
            with self.assertRaisesRegex(ValueError, "Raw-history leak"):
                audit_manifests(root / "train.jsonl", root / "valid.jsonl")

    def test_rejects_real_student_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = [record(i, task, "train") for i, task in enumerate(TASKS)]
            valid = [record(i, task, "valid") for i, task in enumerate(TASKS)]
            train[0]["student_id"] = valid[0]["student_id"] = "real-student"
            write(root / "train.jsonl", train)
            write(root / "valid.jsonl", valid)
            with self.assertRaisesRegex(ValueError, "real-student overlap"):
                audit_manifests(root / "train.jsonl", root / "valid.jsonl")


if __name__ == "__main__":
    unittest.main()
