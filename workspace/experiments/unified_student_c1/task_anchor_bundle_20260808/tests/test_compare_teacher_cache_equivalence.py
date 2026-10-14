from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_teacher_cache_equivalence import compare  # noqa: E402


def write_cache(root: Path, ids: torch.Tensor, values: torch.Tensor) -> None:
    example_id = "x"
    payload = {
        example_id: {
            "example_id": example_id,
            "target_token_ids": [2, 3],
            "teacher_topk_ids": ids,
            "teacher_topk_logprobs": values,
        }
    }
    torch.save(payload, root / "shard-00000.pt")
    (root / "index.json").write_text(
        json.dumps([{"example_id": example_id, "file": "shard-00000.pt"}]),
        encoding="utf-8",
    )


class CompareTeacherCacheEquivalenceTest(unittest.TestCase):
    def test_accepts_small_float16_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "a"
            right = Path(directory) / "b"
            left.mkdir()
            right.mkdir()
            ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
            write_cache(left, ids, torch.tensor([[0.0, -1.0], [-2.0, -3.0]], dtype=torch.float16))
            write_cache(right, ids, torch.tensor([[0.0, -1.0], [-2.0, -3.0005]], dtype=torch.float16))
            result = compare(left, right, 1 / 1024)
            self.assertEqual(result["status"], "equivalent")

    def test_rejects_topk_id_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "a"
            right = Path(directory) / "b"
            left.mkdir()
            right.mkdir()
            values = torch.zeros((2, 2), dtype=torch.float16)
            write_cache(left, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), values)
            write_cache(right, torch.tensor([[1, 2], [3, 5]], dtype=torch.int32), values)
            with self.assertRaisesRegex(ValueError, "Top-k ID mismatch"):
                compare(left, right, 1 / 1024)


if __name__ == "__main__":
    unittest.main()
