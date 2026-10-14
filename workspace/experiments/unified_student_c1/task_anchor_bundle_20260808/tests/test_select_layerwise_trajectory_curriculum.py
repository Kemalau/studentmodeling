from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from select_layerwise_trajectory_curriculum import select  # noqa: E402


class SelectLayerwiseTrajectoryCurriculumTest(unittest.TestCase):
    def write_job(
        self, root: Path, branch: str, rank: int, relative: float, specificity: float
    ) -> None:
        job = root / branch
        metrics = job / "metrics"
        c4 = job / "eval" / "c4"
        metrics.mkdir(parents=True)
        c4.mkdir(parents=True)
        run_dir = (
            "/mnt/amulet/studentmodeling/checkpoints/"
            f"layerwise_trajectory_curriculum_{branch}_seed42_n200_20260731"
        )
        (metrics / "trajectory_curriculum_receipt.json").write_text(json.dumps({
            "status": "trajectory_curriculum_complete", "branch": branch,
            "transition_rank": rank, "relative_mse_weight": relative,
            "optimizer_steps": 50, "remote_run_dir": run_dir,
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
        }))
        (metrics / "completion.json").write_text(json.dumps({
            "status": "diagnostic_complete", "micro_steps": 200,
            "causal_pairs_seen": 200,
        }))
        (metrics / "validation_metrics.json").write_text(json.dumps({
            "accuracy_gap_recurrent_minus_direct": 0.01,
        }))
        (metrics / "gradient_audit.json").write_text(json.dumps({
            "layerwise_transition": {"gradient_abs_sum": 2.0},
            "lora": {"parameter_tensors": 0}, "frozen_tensors_with_grad": [],
        }))
        random_rate = 0.02
        (c4 / "recurrent_causal_gate.json").write_text(json.dumps({
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "summary": {
                "clean_n": 80, "D_specific": specificity, "D_damage": random_rate,
                "clean": {
                    "swap": {"flip_to_A_rate": specificity + random_rate},
                    "random": {"flip_to_A_rate": random_rate},
                    "noop": {"flip_to_A_rate": 0.0},
                    "samemal": {"flip_to_A_rate": 0.03},
                },
            },
        }))

    def test_selects_highest_specificity_passing_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_job(root, "r64_scale", 64, 1.0, 0.16)
            self.write_job(root, "r256_scale", 256, 1.0, 0.24)
            self.write_job(root, "r256_cosine", 256, 0.0, 0.18)
            value = select(root)
            self.assertTrue(value["scientific_gate_pass"])
            self.assertEqual(value["selected_branch"], "r256_scale")
            self.assertEqual(
                value["selected_final_fit_policy"]["causal_objective_mode"],
                "specificity_triplet",
            )

    def test_fails_closed_below_specificity_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_job(root, "r64_scale", 64, 1.0, 0.10)
            self.write_job(root, "r256_scale", 256, 1.0, 0.11)
            self.write_job(root, "r256_cosine", 256, 0.0, 0.12)
            value = select(root)
            self.assertFalse(value["scientific_gate_pass"])
            self.assertIsNone(value["selected_branch"])


if __name__ == "__main__":
    unittest.main()
