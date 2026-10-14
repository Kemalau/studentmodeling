from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m4_causal_rehearsal import SPECS, load, render
from validate_m4_causal_rehearsal_amlt_config import validate


class CausalRehearsalConfigTest(unittest.TestCase):
    def test_rendered_jobs_pass_fail_closed_validator(self) -> None:
        value = render(load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"), list(SPECS))
        validate(value)
        self.assertEqual({job["name"] for job in value["jobs"]}, {x["job"] for x in SPECS.values()})

    def test_sparse_or_resumed_main_schedule_is_rejected(self) -> None:
        value = render(load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"), ["repeat100_w1"])
        command = value["jobs"][0]["command"]
        value["jobs"][0]["command"] = command.replace("--causal-repeat-fraction 1.0", "--causal-repeat-fraction 0.25")
        with self.assertRaisesRegex(ValueError, "Command drift"):
            validate(value)

    def test_undefined_inherited_checkpoint_is_rejected(self) -> None:
        value = render(load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"), ["repeat100_w1"])
        value["jobs"][0]["command"] += '\npython - "$$CHECKPOINT"\n'
        with self.assertRaisesRegex(ValueError, "Forbidden path/protocol"):
            validate(value)

    def test_uai_drift_is_rejected(self) -> None:
        value = render(load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"), ["repeat100_w4"])
        value["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "wrong"
        with self.assertRaisesRegex(ValueError, "UAI drift"):
            validate(value)


if __name__ == "__main__":
    unittest.main()
