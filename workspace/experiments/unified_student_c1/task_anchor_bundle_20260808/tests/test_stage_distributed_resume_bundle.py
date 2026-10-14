from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage_distributed_resume_bundle import stage_bundle, validate_bundle  # noqa: E402


class StageDistributedResumeBundleTest(unittest.TestCase):
    def source_bundle(self, root: Path) -> tuple[Path, dict]:
        source = root / "source/checkpoint-round-750"
        adapter = source / "adapter"
        adapter.mkdir(parents=True)
        state = source / "recurrent_state.pt"
        state.write_bytes(b"state-payload")
        model = adapter / "adapter_model.safetensors"
        model.write_bytes(b"adapter-payload")
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {
            "status": "complete",
            "round": 750,
            "world_size": 16,
            "sha256": {
                "adapter_model": digest(model),
                "recurrent_state": digest(state),
            },
        }
        (source / "checkpoint_completion.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return state, receipt

    def test_stages_complete_hash_valid_bundle_and_repairs_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, receipt = self.source_bundle(root)
            destination = root / "local"
            self.assertEqual(
                stage_bundle(
                    source,
                    destination,
                    expected_state_sha256=receipt["sha256"]["recurrent_state"],
                ),
                receipt,
            )
            self.assertEqual(validate_bundle(destination), receipt)
            self.assertTrue((destination / "recurrent_state.complete").is_file())
            (destination / "recurrent_state.pt").write_bytes(b"corrupt")
            self.assertEqual(stage_bundle(source, destination), receipt)
            self.assertEqual(validate_bundle(destination), receipt)

    def test_expected_source_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.source_bundle(root)
            with self.assertRaisesRegex(ValueError, "source checkpoint"):
                stage_bundle(source, root / "local", expected_state_sha256="0" * 64)

    def test_corrupt_source_is_rejected_after_single_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.source_bundle(root)
            source.write_bytes(b"corrupt-source")
            with self.assertRaisesRegex(ValueError, "hash verification"):
                stage_bundle(source, root / "local")


if __name__ == "__main__":
    unittest.main()
