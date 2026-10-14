from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LocalRecoveryWatchersTest(unittest.TestCase):
    def test_seed97_aggregator_is_fail_closed(self) -> None:
        source = (ROOT / "scripts/watch_promoted_round1000_local_seed97.sh").read_text()
        self.assertIn("{13,42,97}", source)
        self.assertIn("checkpoint_round']==1000", source)
        self.assertIn("two_of_three_gate_pass", source)
        self.assertIn("local_seed97_needs_attention", source)
        self.assertIn('rm -f "$state_dir/needs_attention"', source)
        self.assertIn('touch "$state_dir/complete"', source)

    def test_balanced_t6_downloads_hashes_and_uses_full_support(self) -> None:
        source = (ROOT / "scripts/watch_balanced_lr5e7_round4500_local_t6.sh").read_text()
        self.assertIn("checkpoint-round-4500", source)
        self.assertIn("receipt=={'status':'complete','round':4500,'world_size':4,'sha256':observed}", source)
        self.assertIn("--tasks predict_error_persistence", source)
        self.assertIn("assert support==976", source)
        self.assertIn("local_gpu_readonly_preflight.lock", source)

    def test_round1250_local_seed42_is_exact_and_resumable(self) -> None:
        source = (ROOT / "scripts/watch_promoted_round1250_local_seed42.sh").read_text()
        self.assertIn("receipt=={'status':'complete','round':1250,'world_size':16,'sha256':observed}", source)
        self.assertIn("c4_stage_a_gate_pairs_seed42_n150_v1.json", source)
        self.assertIn("--num-pairs 150 --seed 42", source)
        self.assertIn("--progress-dir", source)
        self.assertIn("local_gpu_readonly_preflight.lock", source)
        self.assertIn("raw_history", (ROOT / "scripts/evaluate_recurrent_causal_gate.py").read_text())


if __name__ == "__main__":
    unittest.main()
