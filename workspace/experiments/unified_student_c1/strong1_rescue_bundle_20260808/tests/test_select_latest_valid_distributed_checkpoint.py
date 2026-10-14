from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from select_latest_valid_distributed_checkpoint import select  # noqa: E402


class SelectLatestValidDistributedCheckpointTest(unittest.TestCase):
    def write_checkpoint(
        self, root: Path, round_index: int, *, planned_optimizer_steps: int = 16_241
    ) -> Path:
        checkpoint = root / f"checkpoint-round-{round_index}"
        adapter = checkpoint / "adapter/adapter_model.safetensors"
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(f"adapter-{round_index}".encode())
        state = checkpoint / "recurrent_state.pt"
        torch.save(
            {"config": {"planned_optimizer_steps": planned_optimizer_steps}},
            state,
        )
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {
            "status": "complete",
            "round": round_index,
            "world_size": 16,
            "sha256": {
                "adapter_model": digest(adapter),
                "recurrent_state": digest(state),
            },
        }
        (checkpoint / "checkpoint_completion.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return checkpoint

    def test_selects_newest_hash_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_checkpoint(root, 250)
            newest = self.write_checkpoint(root, 500)
            self.assertEqual(select(root), newest / "recurrent_state.pt")

    def test_corrupt_newest_falls_back_to_previous_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = self.write_checkpoint(root, 250)
            newest = self.write_checkpoint(root, 500)
            (newest / "recurrent_state.pt").write_bytes(b"corrupt")
            self.assertEqual(select(root), previous / "recurrent_state.pt")

    def test_incomplete_newest_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = self.write_checkpoint(root, 250)
            incomplete = root / "checkpoint-round-500"
            incomplete.mkdir()
            (incomplete / "recurrent_state.pt.partial").write_bytes(b"partial")
            self.assertEqual(select(root), previous / "recurrent_state.pt")

    def test_no_valid_checkpoint_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_checkpoint(root, 250)
            receipt = root / "checkpoint-round-250/checkpoint_completion.json"
            value = json.loads(receipt.read_text())
            value["world_size"] = 8
            receipt.write_text(json.dumps(value))
            self.assertIsNone(select(root))

    def test_out_of_epoch_or_off_cadence_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = self.write_checkpoint(root, 250)
            self.write_checkpoint(root, 251)
            self.write_checkpoint(root, 16_500)
            self.assertEqual(select(root), previous / "recurrent_state.pt")

    def test_recovery_cadence_accepts_hash_valid_emergency_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            emergency = self.write_checkpoint(root, 760)
            self.assertIsNone(select(root))
            self.assertEqual(
                select(root, recovery_save_steps=10),
                emergency / "recurrent_state.pt",
            )

    def test_direct_cadence_probe_survives_stale_blob_listing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.write_checkpoint(root, 25)
            with patch.object(Path, "glob", return_value=[]):
                self.assertIsNone(
                    select(
                        root,
                        expected_world_size=16,
                        max_round=250,
                        save_steps=25,
                        recovery_save_steps=25,
                    )
                )
                self.assertEqual(
                    select(
                        root,
                        expected_world_size=16,
                        max_round=250,
                        save_steps=25,
                        recovery_save_steps=25,
                        probe_expected_cadence=True,
                    ),
                    checkpoint / "recurrent_state.pt",
                )

    def test_recipe_mismatched_checkpoint_is_skipped_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compatible = self.write_checkpoint(
                root, 250, planned_optimizer_steps=1_250
            )
            self.write_checkpoint(root, 500, planned_optimizer_steps=800)
            self.assertEqual(
                select(
                    root,
                    expected_world_size=16,
                    max_round=1_250,
                    save_steps=250,
                    expected_planned_optimizer_steps=1_250,
                ),
                compatible / "recurrent_state.pt",
            )


if __name__ == "__main__":
    unittest.main()
