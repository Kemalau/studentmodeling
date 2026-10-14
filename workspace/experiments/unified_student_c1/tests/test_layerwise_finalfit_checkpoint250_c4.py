from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_layerwise_finalfit_checkpoint_c4_amlt import load, validate
from render_layerwise_finalfit_checkpoint_c4 import render


class LayerwiseFinalFitCheckpointC4Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load(ROOT / "amlt/feeds_layerwise_finalfit_checkpoint250_c4_20260801.yaml")

    def test_config_passes(self) -> None:
        validate(self.config)

    def test_round500_render_passes_the_same_fail_closed_validator(self) -> None:
        value = render(self.config, 500)
        validate(value, 500)
        command = value["jobs"][0]["command"]
        self.assertIn("checkpoint-round-500", command)
        self.assertIn("--expected-round 500", command)
        self.assertIn("adapter/adapter_config.json", command)

    def test_frozen_training_runtime_is_required(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py",
            "scripts/layerwise_recurrent_student_model.py",
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(value)

    def test_scalar_drift_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "--state-architecture layerwise", "--state-architecture scalar"
        )
        with self.assertRaises(ValueError):
            validate(value)

    def test_wrong_world_size_is_rejected(self) -> None:
        value = copy.deepcopy(self.config)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "receipt.get('world_size') == 16", "receipt.get('world_size') == 4"
        )
        with self.assertRaises(ValueError):
            validate(value)

    def test_watcher_waits_for_atomic_checkpoint_and_recovers(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_finalfit_checkpoint250_c4.sh").read_text(encoding="utf-8")
        for required in (
            "adapter/adapter_config.json adapter/adapter_model.safetensors recurrent_state.pt checkpoint_completion.json",
            "for recovery_wave in 0 1 2 3",
            "terminal_status.txt",
            "formal_final_c4_required",
            "feeds_submission_global.lock",
            "snapshot_prefix=",
            "snapshot_root=",
            "copy_blob_verified",
            "properties.copy.status",
            "--source-if-match",
            "copy timed out",
            "snapshot_checkpoint_completion.json",
            "snapshot_complete",
            "contentLength",
            "adapter/adapter_config.json",
            "copy_status in (None, 'success')",
            '[[ -f "$auth_required" ]]',
            "while ! az account get-access-token",
            "checkpoint_resume_audit.json",
            "exact_resume_ready",
            "attention_is_current",
            "No jobs found in the current experiment",
            'active_experiment.txt" -nt "$marker"',
            '"$active_pid" -nt "$marker"',
        ):
            self.assertIn(required, text)
        self.assertIn('[[ "$checkpoint_round" == 500 || "$checkpoint_round" == 1000 ]]', text)
        self.assertNotIn("--requires-sync true", text)
        config = (
            ROOT / "amlt/feeds_layerwise_finalfit_checkpoint250_c4_20260801.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("c4_eval_progress_v1", config)
        self.assertIn("--progress-dir", config)
        self.assertIn("--progress-key", config)
        self.assertNotIn("--file', '/dev/stdout'", text)


if __name__ == "__main__":
    unittest.main()
