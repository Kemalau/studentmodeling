from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_checkpoint500_causal_rescue import summarize  # noqa: E402


class RescueSummaryTest(unittest.TestCase):
    def write_arm(self, root: Path, branch: str, causal_pairs: int, repeat: float, weight: float, d_specific: float) -> None:
        arm = root / branch; arm.mkdir(parents=True)
        values = {
            "rescue_training_receipt.json": {"branch":branch,"diagnostic_only":True,"exact_resume":False,"optimizer_reset":True,"source_checkpoint_round":500,"source_world_size":16,"current_world_size":4,"start_global_position":8000,"end_global_position":8999},
            "completion.json": {"status":"diagnostic_complete","micro_steps":1000,"optimizer_steps":250,"history_rows":1000,"history_global_position_start":8000,"history_global_position_end":8999,"causal_pairs_seen":causal_pairs},
            "run_config.json": {"diagnostic_optimizer_reset":True,"causal_repeat_fraction":repeat,"causal_weight":weight,"causal_objective_mode":"specificity_triplet"},
            "recurrent_causal_gate.json": {"state_architecture":"layerwise","state_shape":[64,5120],"raw_history_visible_to_decoder":False,"persistent_history_kv_cache":False,"summary":{"all_pairs_n":150,"clean_n":80,"D_specific":d_specific,"scientific_gate_pass":d_specific>=0.15,"clean":{"swap":{"flip_to_A_rate":0.3},"random":{"flip_to_A_rate":0.3-d_specific},"samemal":{"flip_to_A_rate":0.05},"noop":{"flip_to_A_rate":0.0}}}},
        }
        for name,value in values.items(): (arm/name).write_text(json.dumps(value))

    def test_selects_best_passing_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.write_arm(root,"dense_same_total",100,0.10,1.0,0.16)
            self.write_arm(root,"strong_same_density",50,0.05,2.0,0.20)
            value=summarize(root)
            self.assertTrue(value["any_scientific_gate_pass"])
            self.assertEqual(value["selected_branch"],"strong_same_density")

    def test_branch_identity_comes_from_receipt_not_job_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_arm(root / "amlt-job-with-hyphens", "dense_same_total", 100, 0.10, 1.0, 0.16)
            self.write_arm(root / "another-job-name", "strong_same_density", 50, 0.05, 2.0, 0.20)
            value = summarize(root)
            self.assertEqual(value["selected_branch"], "strong_same_density")


if __name__ == "__main__": unittest.main()
