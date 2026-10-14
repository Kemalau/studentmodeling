from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchPromotedTrajectoryFromC4Test(unittest.TestCase):
    def test_joint_screen_is_provenance_pinned_and_serialized(self) -> None:
        source = (ROOT / "scripts/watch_promoted_checkpoint_trajectory_from_c4.sh").read_text()
        for fragment in (
            "CHECKPOINT_ROUND",
            "SOURCE_C4_SUBDIR",
            "TRAJECTORY_SUBDIR",
            "artifact_audit.json",
            "exact_resume_audit.json",
            "promoted_recipe_audit.json",
            "local_gpu_readonly_preflight.lock",
            "predict_correctness predict_error_persistence diagnose_misconception",
            "judge_formal_t3_predictions.py",
            "validate_checkpoint_t3_semantic_result.py",
            "summarize_promoted_checkpoint_trajectory.py",
            "diagnostic_gate_failed",
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("amlt run", source)


if __name__ == "__main__":
    unittest.main()
