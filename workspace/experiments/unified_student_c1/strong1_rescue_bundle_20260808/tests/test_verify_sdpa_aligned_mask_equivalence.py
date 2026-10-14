from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_sdpa_aligned_mask_equivalence import verify  # noqa: E402


class AlignedSdpaMaskEquivalenceTest(unittest.TestCase):
    def test_qwen3_549_token_value_loss_and_gradient_equivalence(self) -> None:
        value = verify()
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["tokens"], 549)
        self.assertEqual(value["mask_shape"], [1, 1, 549, 549])
        self.assertEqual(value["mask_stride"][1] % 4, 0)
        self.assertLessEqual(value["max_gradient_abs_diff"], 2e-7)

    def test_qwen3_779_token_checkpoint_offload_matches_native_gradient(self) -> None:
        value = verify(779, checkpoint_offload=True)
        self.assertEqual(value["status"], "passed")
        self.assertEqual(value["tokens"], 779)
        self.assertTrue(value["checkpoint_offload"])
        self.assertEqual(value["mask_shape"], [1, 1, 779, 779])
        self.assertEqual(value["mask_stride"][1] % 4, 0)
        self.assertLessEqual(value["max_gradient_abs_diff"], 2e-7)


if __name__ == "__main__":
    unittest.main()
