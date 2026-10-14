from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_m4_diagnostic_result import validate  # noqa: E402


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M4DiagnosticResultTest(unittest.TestCase):
    def fixture(self, projection: str = "trainable") -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        write(
            root / "gradient_audit.json",
            {
                "frozen_tensors_with_grad": [],
                "lora": {
                    "parameter_tensors": 4,
                    "tensors_with_nonzero_gradient": 4,
                    "gradient_abs_sum": 2.0,
                },
                "state_projection": {
                    "parameter_tensors": 4 if projection == "trainable" else 0,
                    "tensors_with_nonzero_gradient": 2 if projection == "trainable" else 0,
                    "gradient_abs_sum": 1.0 if projection == "trainable" else 0.0,
                },
            },
        )
        write(root / "validation_metrics.json", {"candidate_n": 20, "accuracy": 0.5})
        write(
            root / "causal_validation_metrics.json",
            {"n": 20, "clean_n": 10, "native_accuracy": 0.5},
        )
        write(
            root / "run_config.json",
            {
                "trainable_projections": projection == "trainable",
                "behavior_kd_weight": 0.5,
            },
        )
        hashes = {
            name: digest(root / f"{name}.json")
            for name in (
                "validation_metrics",
                "causal_validation_metrics",
                "run_config",
                "gradient_audit",
            )
        }
        write(
            root / "completion.json",
            {
                "status": "diagnostic_complete",
                "diagnostic": True,
                "micro_steps": 4,
                "optimizer_steps": 1,
                "real_examples_seen": 4,
                "causal_pairs_seen": 1,
                "contains_causal_pair": True,
                "sha256": hashes,
            },
        )
        return temporary, root

    def test_valid_trainable_kd_diagnostic_passes(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        self.assertTrue(validate(root, "trainable", "generation_kd")["pass"])

    def test_hash_or_missing_causal_path_fails(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        completion = json.loads((root / "completion.json").read_text())
        completion["contains_causal_pair"] = False
        write(root / "completion.json", completion)
        with self.assertRaisesRegex(ValueError, "causal_path_exercised"):
            validate(root, "trainable", "generation_kd")

    def test_mode_mismatch_fails(self) -> None:
        temporary, root = self.fixture()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(ValueError, "projection_mode_matches"):
            validate(root, "identity", "generation_kd")


if __name__ == "__main__":
    unittest.main()
