from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_finalfit_cache_v2_amlt_config import load_config, validate  # noqa: E402


CONFIG = ROOT / "amlt/feeds_recurrent_finalfit_cache_v2_16way_20260731.yaml"


class FinalFitCacheV2AMLTConfigTest(unittest.TestCase):
    def test_exact_template_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_state_token_alias_is_not_a_template_placeholder(self) -> None:
        config = load_config(CONFIG)
        self.assertIn("<|vision_pad|>", config["jobs"][0]["command"])
        validate(config)

    def test_selected_failed_rank_subset_passes_preflight(self) -> None:
        config = load_config(CONFIG)
        selected = {
            "finalfit-cache-v2-rank02-seed42",
            "finalfit-cache-v2-rank11-seed42",
        }
        config["jobs"] = [job for job in config["jobs"] if job["name"] in selected]
        validate(config, selected)

    def test_partition_and_numeric_path_drift_fail(self) -> None:
        config = load_config(CONFIG)
        partition = copy.deepcopy(config)
        partition["jobs"][7]["submit_args"]["env"]["PARTITION_INDEX"] = "6"
        with self.assertRaisesRegex(ValueError, "partition drift"):
            validate(partition)
        numeric = copy.deepcopy(config)
        numeric["jobs"][0]["command"] = numeric["jobs"][0]["command"].replace(
            "--attn-implementation eager", "--attn-implementation sdpa"
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(numeric)

    def test_gpu_and_uai_drift_fail(self) -> None:
        config = load_config(CONFIG)
        gpu = copy.deepcopy(config)
        gpu["jobs"][0]["sku"] = "80G4-A100"
        with self.assertRaisesRegex(ValueError, "GPU/MPI drift"):
            validate(gpu)
        uai = copy.deepcopy(config)
        uai["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "other"
        with self.assertRaisesRegex(ValueError, "Identity/partition drift"):
            validate(uai)


if __name__ == "__main__":
    unittest.main()
