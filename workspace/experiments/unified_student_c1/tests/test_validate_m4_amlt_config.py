from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_m4_amlt_config import (  # noqa: E402
    CACHE_JOB,
    TRAIN_SCRIPT_SHA256,
    load_config,
    selected_job_name,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "amlt/feeds_recurrent_student_m4_full_20260730.yaml"


class M4ConfigTest(unittest.TestCase):
    def test_full_template_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_selected_cache_and_diagnostic_pass(self) -> None:
        config = load_config(CONFIG)
        selected = {
            CACHE_JOB,
            selected_job_name("diagnostic", "generation_kd", "trainable"),
        }
        config["jobs"] = [job for job in config["jobs"] if job["name"] in selected]
        validate(config, selected_names=selected)

    def test_job_label_and_target_drift_fail(self) -> None:
        config = load_config(CONFIG)
        label = copy.deepcopy(config)
        label["jobs"][1]["submit_args"]["env"]["RUN_KIND"] = "full"
        with self.assertRaisesRegex(ValueError, "label drift"):
            validate(label)
        target = copy.deepcopy(config)
        target["target"]["name"] = "Other"
        with self.assertRaisesRegex(ValueError, "Target drift"):
            validate(target)

    def test_training_script_hash_drift_fails(self) -> None:
        config = load_config(CONFIG)
        job = next(
            item for item in config["jobs"]
            if item["name"].startswith("m4-full-generation-only-identity")
        )
        self.assertIn(TRAIN_SCRIPT_SHA256, job["command"])
        job["command"] = job["command"].replace(TRAIN_SCRIPT_SHA256, "0" * 64)
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(config)

    def test_diagnostic_support_limit_drift_fails(self) -> None:
        config = load_config(CONFIG)
        job = next(item for item in config["jobs"] if item["name"].startswith("m4-diagnostic-generation-only-identity"))
        job["command"] = job["command"].replace("--eval-limit 40", "--eval-limit 20")
        with self.assertRaisesRegex(ValueError, "Diagnostic limit"):
            validate(config)

    def test_memory_diagnostic_is_single_gpu_and_long_enough(self) -> None:
        config = load_config(CONFIG)
        job = next(
            item
            for item in config["jobs"]
            if item["name"] == "m4-memory-diagnostic-generation-kd-trainable-seed42"
        )
        self.assertEqual(job["sku"], "80G1-A100")
        self.assertEqual(job["submit_args"]["env"]["DIAGNOSTIC_STEPS"], "200")
        selected = {job["name"]}
        config["jobs"] = [job]
        validate(config, selected_names=selected)

    def test_full_jobs_are_four_gpu_mpi_and_diagnostics_remain_single_gpu(self) -> None:
        config = load_config(CONFIG)
        for job in config["jobs"]:
            if job["name"].startswith("m4-full-generation-"):
                self.assertEqual(job["sku"], "80G4-A100")
                self.assertIs(job["mpi"], True)
                self.assertEqual(job["process_count_per_node"], -1)
                self.assertIn("--distributed-mpi", job["command"])
            elif job["name"].startswith("m4-diagnostic-"):
                self.assertEqual(job["sku"], "80G1-A100")


if __name__ == "__main__":
    unittest.main()
