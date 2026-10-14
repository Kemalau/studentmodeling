from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_final_fit_amlt_config import (  # noqa: E402
    CACHE_JOBS,
    load_config,
    selected_train_job,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_recurrent_student_final_fit_20260730.yaml"


class FinalFitAMLTConfigTest(unittest.TestCase):
    def test_full_template_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_selected_generation_only_job_passes_without_cache(self) -> None:
        config = load_config(CONFIG)
        name = selected_train_job("generation_only", "trainable")
        config["jobs"] = [job for job in config["jobs"] if job["name"] == name]
        validate(config, {name})

    def test_selected_kd_jobs_pass_with_all_cache_partitions(self) -> None:
        config = load_config(CONFIG)
        train = selected_train_job("generation_kd", "identity")
        selected = set(CACHE_JOBS) | {train}
        config["jobs"] = [job for job in config["jobs"] if job["name"] in selected]
        validate(config, selected)

    def test_selected_sixteen_rank_job_passes_with_equivalent_global_batch(self) -> None:
        config = load_config(CONFIG)
        name = "finalfit-generation-kd-trainable-seed42-16rank"
        config["jobs"] = [job for job in config["jobs"] if job["name"] == name]
        validate(config, {name})
        job = config["jobs"][0]
        self.assertEqual(job["sku"], "2x80G8-A100-IB-NvLink")
        self.assertEqual(job["submit_args"]["env"]["DISTRIBUTED_WORLD_SIZE"], "16")
        self.assertEqual(job["submit_args"]["env"]["GRADIENT_ACCUMULATION_STEPS"], "1")

    def test_dump_style_single_dollar_receipt_copy_passes(self) -> None:
        config = load_config(CONFIG)
        name = "finalfit-generation-kd-trainable-seed42-16rank"
        config["jobs"] = [job for job in config["jobs"] if job["name"] == name]
        command = config["jobs"][0]["command"]
        config["jobs"][0]["command"] = command.replace(
            'cp "$$PILOT_RECEIPT" "$$AMLT_OUTPUT_DIR/metrics/pilot_receipt.json"',
            'cp "$PILOT_RECEIPT" "$AMLT_OUTPUT_DIR/metrics/pilot_receipt.json"',
        )
        validate(config, {name})

    def test_missing_receipt_copy_fails(self) -> None:
        config = load_config(CONFIG)
        name = "finalfit-generation-kd-trainable-seed42-16rank"
        config["jobs"] = [job for job in config["jobs"] if job["name"] == name]
        config["jobs"][0]["command"] = config["jobs"][0]["command"].replace(
            'cp "$$PILOT_RECEIPT" "$$AMLT_OUTPUT_DIR/metrics/pilot_receipt.json"',
            ": # removed receipt copy",
        )
        with self.assertRaisesRegex(ValueError, "Pilot-receipt copy drift"):
            validate(config, {name})

    def test_final_fit_inherits_causal_policy_from_validated_m4_receipt(self) -> None:
        config = load_config(CONFIG)
        name = selected_train_job("generation_kd", "trainable")
        job = next(job for job in config["jobs"] if job["name"] == name)
        command = job["command"]
        self.assertIn("PILOT_CAUSAL_WEIGHT", command)
        self.assertIn("PILOT_NORMALIZE_CAUSAL", command)
        self.assertIn("PILOT_CAUSAL_REPEAT", command)
        self.assertIn("PILOT_CAUSAL_OBJECTIVE_MODE", command)
        self.assertIn("causal_train_sha256", command)
        self.assertIn("causal_valid_sha256", command)
        self.assertIn("causal_args=(--causal-weight", command)
        self.assertIn("--normalize-causal-exposure", command)
        self.assertIn("--causal-objective-mode", command)
        self.assertIn("C4_D_specific", command)
        self.assertNotIn("--causal-weight 1.0", command)

    def test_gpu_replay_and_hash_drift_fail(self) -> None:
        config = load_config(CONFIG)
        train = next(job for job in config["jobs"] if job["name"].startswith("finalfit-generation-only-identity"))
        gpu = copy.deepcopy(config)
        next(job for job in gpu["jobs"] if job["name"] == train["name"])["sku"] = "80G1-A100"
        with self.assertRaisesRegex(ValueError, "GPU/MPI drift"):
            validate(gpu)
        replay = copy.deepcopy(config)
        target = next(job for job in replay["jobs"] if job["name"] == train["name"])
        target["command"] = target["command"].replace(
            '--causal-repeat-fraction "$$PILOT_CAUSAL_REPEAT"',
            "--causal-repeat-fraction 0.01",
        )
        with self.assertRaisesRegex(ValueError, "command drift"):
            validate(replay)
        digest = copy.deepcopy(config)
        target = next(job for job in digest["jobs"] if job["name"] == train["name"])
        target["command"] = target["command"].replace(
            "865c12219deb23cf9db145c797244969721fce80323050dcd987b33d96cbfb1b",
            "0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(digest)


if __name__ == "__main__":
    unittest.main()
