from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_task_retention_promoted_final_fit_amlt import (  # noqa: E402
    load,
    validate,
)


class PromotedFinalFitAmltTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(
            ROOT / "amlt/feeds_layerwise_task_retention_promoted_final_fit_20260801.yaml"
        )

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_launcher_forces_the_promoted_job_venv(self) -> None:
        job = self.config["jobs"][0]
        self.assertEqual(
            job["submit_args"]["env"]["VENV_ROOT"],
            "/tmp/studentmodeling-layerwise-promoted-final-venv",
        )
        launcher = (ROOT / "scripts/run_layerwise_task_retention_promoted_final_fit.sh").read_text()
        self.assertIn('python() { "$VENV_ROOT/bin/python" "$@"; }', launcher)
        self.assertIn("import accelerate,bitsandbytes,peft,safetensors,transformers", launcher)
        self.assertIn("PROMOTION_LOSS_PROFILE", launcher)
        self.assertIn("state_path) BEHAVIOR_KD_WEIGHT=0.0", launcher)
        self.assertIn("STUDENTMODELING_TEACHER_CACHE_UNUSED=1", launcher)
        self.assertIn("train_recurrent_student_m4_promoted.py", launcher)
        self.assertIn(
            "INITIAL_CHECKPOINT_SHA256=598901f8b8fc0901c6a1da6eaa32866dc1fd7dbfe762d34483b10dbaf23853fe",
            launcher,
        )
        self.assertIn('/tmp/layerwise-promoted-resume/$resume_key', launcher)
        self.assertIn("stage_distributed_resume_bundle.py", launcher)
        self.assertIn('stage_args=(--source-state "$resume_checkpoint"', launcher)
        self.assertIn(
            'resume_args=(--resume-from-checkpoint "$local_resume")', launcher
        )
        self.assertIn("--recovery-save-steps 10", launcher)
        self.assertIn("--recovery-checkpoint-first-offset 10", launcher)
        self.assertIn("--recovery-checkpoint-interval 20", launcher)

    def test_gpu_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["sku"] = "80G8-A100-IB-NvLink"
        with self.assertRaisesRegex(ValueError, "GPU"):
            validate(value)

    def test_launcher_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = "set -euo pipefail\nbash scripts/wrong.sh\n"
        with self.assertRaisesRegex(ValueError, "launcher"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
