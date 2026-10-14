from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_multinode_dist_diag import validate  # noqa: E402


class MultinodeDistDiagTest(unittest.TestCase):
    def fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        for rank in range(16):
            row = {
                "status": "pass", "rank": rank, "local_rank": rank % 8,
                "world_size": 16, "hostname": f"node-{rank // 8}",
                "master_address": "node-0", "all_reduce_sum": 136.0,
                "device": "NVIDIA A100-SXM4-80GB",
            }
            (root / f"rank-{rank}.json").write_text(json.dumps(row))
        return temporary, root

    def test_complete_two_node_receipt_passes(self) -> None:
        temporary, root = self.fixture(); self.addCleanup(temporary.cleanup)
        self.assertTrue(validate(root)["pass"])

    def test_duplicate_host_or_wrong_sum_fails(self) -> None:
        temporary, root = self.fixture(); self.addCleanup(temporary.cleanup)
        path = root / "rank-15.json"; row = json.loads(path.read_text())
        row["hostname"] = "node-0"; row["all_reduce_sum"] = 135.0
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "two_hosts|eight_local|all_reduce"):
            validate(root)


if __name__ == "__main__":
    unittest.main()
