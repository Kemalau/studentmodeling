from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from promoted_checkpoint_cadence import (  # noqa: E402
    should_save_distributed_checkpoint,
)


class PromotedRecoveryCheckpointCadenceTest(unittest.TestCase):
    def due(self, round_index: int, *, start: int = 750) -> bool:
        return should_save_distributed_checkpoint(
            round_index,
            start_round=start,
            total_rounds=16_241,
            save_steps=250,
            recovery_first_offset=10,
            recovery_interval=20,
        )

    def test_writes_early_and_sparse_recovery_checkpoints(self) -> None:
        self.assertFalse(self.due(750))
        self.assertTrue(self.due(760))
        self.assertFalse(self.due(761))
        self.assertTrue(self.due(780))
        self.assertTrue(self.due(960))
        self.assertTrue(self.due(1_000))

    def test_cadence_reanchors_after_resume(self) -> None:
        self.assertTrue(self.due(770, start=760))
        self.assertTrue(self.due(970, start=760))
        self.assertTrue(self.due(1_000, start=760))

    def test_final_round_is_always_saved(self) -> None:
        self.assertTrue(self.due(16_241))


if __name__ == "__main__":
    unittest.main()
