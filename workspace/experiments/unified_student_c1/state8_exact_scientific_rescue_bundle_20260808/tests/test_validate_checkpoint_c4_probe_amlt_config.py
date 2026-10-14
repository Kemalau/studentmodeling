from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_checkpoint_c4_probe_amlt_config import load_config, validate  # noqa: E402
from render_checkpoint_c4_probe import render  # noqa: E402

CONFIG = ROOT / "amlt/feeds_m4_checkpoint_c4_probe_20260731.yaml"


class CheckpointC4ProbeConfigTest(unittest.TestCase):
    def test_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_checkpoint_and_uai_drift_fail(self) -> None:
        config = load_config(CONFIG)
        checkpoint = copy.deepcopy(config)
        checkpoint["jobs"][0]["command"] = checkpoint["jobs"][0]["command"].replace("checkpoint-round-250", "checkpoint-round-251")
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(checkpoint)
        uai = copy.deepcopy(config)
        uai["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "UAI drift"):
            validate(uai)

    def test_round_1000_render_passes_dynamic_validation(self) -> None:
        rendered = render(load_config(CONFIG), 1000)
        validate(rendered, 1000)
        job = rendered["jobs"][0]
        self.assertEqual(job["name"], "m4-checkpoint1000-c4-seed42")
        self.assertIn("checkpoint-round-1000", job["command"])
        self.assertIn("checkpoint_round=1000", job["command"])

    def test_custom_remote_run_and_job_render_without_protocol_drift(self) -> None:
        from render_checkpoint_c4_probe import render

        run_dir = (
            "/mnt/amulet/studentmodeling/diagnostics/"
            "world_model_v1_m4_causal_bootstrap_balanced_sum_w1_seed42_round500_20260731"
        )
        job_name = "m4-directional-w1-checkpoint250-c4-seed42"
        rendered = render(
            load_config(ROOT / "amlt/feeds_m4_checkpoint_c4_probe_20260731.yaml"),
            250,
            run_dir=run_dir,
            job_name=job_name,
        )
        validate(rendered, 250, run_dir=run_dir, job_name=job_name)
        command = rendered["jobs"][0]["command"]
        self.assertIn(f"export RUN_DIR={run_dir}", command)
        self.assertIn("--num-pairs 150", command)


if __name__ == "__main__":
    unittest.main()
