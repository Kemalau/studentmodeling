from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_m4_causal_bootstrap_pair8 import load, render  # noqa: E402
from validate_m4_causal_bootstrap_pair8_amlt_config import validate  # noqa: E402


class CausalBootstrapPair8Test(unittest.TestCase):
    def setUp(self) -> None:
        self.value = render(load(ROOT / "amlt/feeds_m4_causal_pair_overfit_20260731.yaml"))

    def test_rendered_job_passes_fail_closed_validator(self) -> None:
        validate(self.value)
        env = self.value["jobs"][0]["submit_args"]["env"]
        self.assertEqual(env["OBJECTIVE_MODE"], "bootstrap_then_symmetric")
        self.assertEqual(env["BOOTSTRAP_NATIVE_WEIGHT"], "0.0")

    def test_native_anchor_variant_is_independent_and_validated(self) -> None:
        anchored = render(
            load(ROOT / "amlt/feeds_m4_causal_pair_overfit_20260731.yaml"),
            "native25",
        )
        validate(anchored, "native25")
        job = anchored["jobs"][0]
        self.assertEqual(
            job["name"], "m4-pair8-overfit-sum-bootstrap-native25-seed42"
        )
        self.assertEqual(job["submit_args"]["env"]["BOOTSTRAP_NATIVE_WEIGHT"], "0.25")
        self.assertEqual(job["submit_args"]["env"]["BRANCH"], "pair8_sum_bootstrap_native25")

    def test_balanced_directional_tail_variants_are_independent_and_validated(self) -> None:
        for variant, native_weight in (("balanced", "0.0"), ("native25_balanced", "0.25")):
            value = render(
                load(ROOT / "amlt/feeds_m4_causal_pair_overfit_20260731.yaml"),
                variant,
            )
            validate(value, variant)
            env = value["jobs"][0]["submit_args"]["env"]
            self.assertEqual(env["OBJECTIVE_MODE"], "bootstrap_then_directional")
            self.assertEqual(env["BOOTSTRAP_NATIVE_WEIGHT"], native_weight)

    def test_token_or_uai_drift_fails(self) -> None:
        drift = copy.deepcopy(self.value)
        drift["jobs"][0]["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] = "wrong"
        with self.assertRaisesRegex(ValueError, "environment drift"):
            validate(drift)

    def test_watcher_recovers_authoritative_checkpoint_before_redundant_copy(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_bootstrap_pair8.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("recover_authoritative_completion", text)
        self.assertIn("final_adapter/adapter_model.safetensors", text)
        self.assertIn("direct_authoritative_recovery.json", text)
        self.assertIn("scientific_gate_failed", text)
        self.assertIn("missed the scientific gate; releasing GPU", text)
        self.assertIn('amlt cancel "$experiment" ":$job" -y', text)


if __name__ == "__main__":
    unittest.main()
