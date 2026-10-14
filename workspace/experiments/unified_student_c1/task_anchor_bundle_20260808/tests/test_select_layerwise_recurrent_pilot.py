from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_layerwise_recurrent_pilot import select  # noqa: E402


class SelectLayerwiseRecurrentPilotTest(unittest.TestCase):
    def write_job(self, root: Path, name: str, d_specific: float) -> None:
        job = root / name
        metrics = job / "metrics"
        c4 = job / "eval" / "c4"
        metrics.mkdir(parents=True)
        c4.mkdir(parents=True)
        (metrics / "pilot_receipt.json").write_text(json.dumps({
            "status": "pilot_training_complete", "branch": name,
            "micro_steps": 200,
            "causal_pairs_seen": 200,
            "learning_rate": 2e-6,
            "causal_weight": 1.0,
            "causal_objective_mode": (
                "specificity_triplet" if name == "specificity" else "donor_directional"
            ),
            "remote_run_dir": (
                "/mnt/amulet/studentmodeling/checkpoints/"
                f"layerwise_recurrent_pilot_{name}_seed42_n200_20260731"
            ),
            "state_architecture": "layerwise",
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
        }))
        (metrics / "validation_metrics.json").write_text(json.dumps({
            "accuracy_gap_recurrent_minus_direct": 0.0,
            "by_task": {
                "diagnose_misconception": {
                    "accuracy_gap_recurrent_minus_direct": 0.0,
                }
            },
        }))
        (metrics / "gradient_audit.json").write_text(json.dumps({"lora": {"gradient_abs_sum": 1.0}}))
        (c4 / "recurrent_causal_gate.json").write_text(json.dumps({"summary": {
            "clean_n": 80, "D_specific": d_specific,
            "clean": {
                "swap": {"flip_to_A_rate": d_specific + 0.1},
                "random": {"flip_to_A_rate": 0.1},
                "samemal": {"flip_to_A_rate": 0.05},
                "noop": {"flip_to_A_rate": 0.0},
            },
        }}))

    def test_selects_best_passing_specificity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_job(root, "preserve_low_lr", 0.16)
            self.write_job(root, "specificity", 0.24)
            self.write_job(root, "balanced", 0.10)
            preserve = root / "preserve_low_lr" / "metrics" / "pilot_receipt.json"
            payload = json.loads(preserve.read_text())
            payload["learning_rate"] = 5e-7
            payload["causal_weight"] = 0.5
            preserve.write_text(json.dumps(payload))
            value = select(root)
            self.assertTrue(value["scientific_gate_pass"])
            self.assertEqual(value["selected_branch"], "specificity")

    def test_fails_when_no_branch_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, score in zip(
                ("preserve_low_lr", "balanced", "specificity"),
                (0.01, 0.02, 0.03),
                strict=True,
            ):
                self.write_job(root, name, score)
            preserve = root / "preserve_low_lr" / "metrics" / "pilot_receipt.json"
            payload = json.loads(preserve.read_text())
            payload["learning_rate"] = 5e-7
            payload["causal_weight"] = 0.5
            preserve.write_text(json.dumps(payload))
            value = select(root)
            self.assertFalse(value["scientific_gate_pass"])
            self.assertIsNone(value["selected_branch"])

    def test_rejects_unmounted_or_wrong_pilot_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("preserve_low_lr", "balanced", "specificity"):
                self.write_job(root, name, 0.2)
            preserve = root / "preserve_low_lr" / "metrics" / "pilot_receipt.json"
            preserve_payload = json.loads(preserve.read_text())
            preserve_payload["learning_rate"] = 5e-7
            preserve_payload["causal_weight"] = 0.5
            preserve.write_text(json.dumps(preserve_payload))
            receipt = root / "balanced" / "metrics" / "pilot_receipt.json"
            value = json.loads(receipt.read_text())
            value["remote_run_dir"] = "/tmp/not-the-mounted-pilot"
            receipt.write_text(json.dumps(value))
            selected = select(root)
            row = next(x for x in selected["candidates"] if x["branch"] == "balanced")
            self.assertFalse(row["checks"]["remote_pilot_artifacts"])
            self.assertFalse(row["pass"])

    def test_rejects_incomplete_or_unregistered_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("preserve_low_lr", "balanced", "specificity"):
                self.write_job(root, name, 0.2)
            preserve = root / "preserve_low_lr" / "metrics" / "pilot_receipt.json"
            preserve_payload = json.loads(preserve.read_text())
            preserve_payload["learning_rate"] = 5e-7
            preserve_payload["causal_weight"] = 0.5
            preserve.write_text(json.dumps(preserve_payload))
            receipt = root / "specificity" / "metrics" / "pilot_receipt.json"
            payload = json.loads(receipt.read_text())
            payload["causal_pairs_seen"] = 199
            receipt.write_text(json.dumps(payload))
            selected = select(root)
            row = next(x for x in selected["candidates"] if x["branch"] == "specificity")
            self.assertFalse(row["checks"]["exposure_complete"])
            self.assertFalse(row["pass"])


if __name__ == "__main__":
    unittest.main()
