from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_m4_causal_finalize_amlt_config import load_config, validate  # noqa: E402


CONFIG = ROOT / "amlt/feeds_m4_causal_strength_finalize_20260731.yaml"


class M4CausalFinalizeConfigTest(unittest.TestCase):
    def test_config_passes(self) -> None:
        validate(load_config(CONFIG))

    def test_selected_job_subset_passes(self) -> None:
        config = load_config(CONFIG)
        selected = {config["jobs"][0]["name"]}
        config["jobs"] = [config["jobs"][0]]
        validate(config, selected)

    def test_mpi_or_checkpoint_drift_fails(self) -> None:
        config = load_config(CONFIG)
        mpi = copy.deepcopy(config)
        mpi["jobs"][0]["mpi"] = True
        with self.assertRaisesRegex(ValueError, "GPU/MPI drift"):
            validate(mpi)
        checkpoint = copy.deepcopy(config)
        checkpoint["jobs"][0]["command"] = checkpoint["jobs"][0]["command"].replace(
            "checkpoint-round-500", "checkpoint-round-499"
        )
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(checkpoint)


if __name__ == "__main__":
    unittest.main()
