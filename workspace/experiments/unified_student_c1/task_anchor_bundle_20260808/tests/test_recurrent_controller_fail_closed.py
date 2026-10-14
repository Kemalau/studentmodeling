from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class RecurrentControllerFailClosedTest(unittest.TestCase):
    def test_causal_finalize_uses_single_gpu_checkpoint_only_path(self) -> None:
        source = (ROOT / "scripts/watch_m4_causal_finalize.sh").read_text(
            encoding="utf-8"
        )
        config = (ROOT / "amlt/feeds_m4_causal_strength_finalize_20260731.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("summarize_m4_causal_strength_branches.py", source)
        self.assertIn("terminal_nonpass_job_names", source)
        self.assertIn("--validation-only-from-distributed-checkpoint", config)
        self.assertNotIn("--distributed-mpi", config)
        self.assertEqual(config.count("sku: 80G1-A100"), 2)

    def test_downstream_controllers_detect_m3_scientific_failure(self) -> None:
        for name in (
            "resume_recurrent_m4_after_m3.sh",
            "resume_recurrent_final_after_m4.sh",
        ):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("m3_scientific_failed=", text)
            self.assertIn('[[ ! -f "$m3_scientific_failed" ]] || fail', text)

    def test_m3_watcher_accepts_only_explicit_strict_or_rescue_gate(self) -> None:
        text = (ROOT / "scripts" / "watch_recurrent_m3_ab.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('value.get("scientific_gate_pass") is True', text)
        self.assertIn('value.get("downstream_rescue_authorized") is True', text)
        self.assertIn('rm -f "$monitor_dir/complete" "$monitor_dir/needs_attention"', text)

    def test_final_controller_recovers_fit_and_evaluation(self) -> None:
        text = (ROOT / "scripts" / "resume_recurrent_final_after_m4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("wait_pass_recoverable()", text)
        self.assertIn("for recovery_wave in 0 1 2 3", text)
        self.assertIn('fit_experiment="${fit_base_experiment}-recovery${recovery_wave}"', text)
        self.assertIn('eval_experiment="${eval_base_experiment}-recovery${recovery_wave}"', text)
        self.assertIn("final fit exhausted three recovery waves", text)
        self.assertIn("evaluation exhausted three recovery waves", text)

    def test_final_controller_prefers_live_sixteen_way_cache(self) -> None:
        text = (ROOT / "scripts" / "resume_recurrent_final_after_m4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("16-way exact teacher cache controller is healthy", text)
        self.assertIn("legacy 4-way cache skipped", text)
        self.assertIn("activating 4-way fallback", text)
        self.assertIn("legacy_cache_experiment=", text)
        self.assertIn("16-way and legacy 4-way teacher caches exhausted recovery waves", text)

    def test_m4_controller_recovers_formal_evaluation(self) -> None:
        text = (ROOT / "scripts" / "resume_recurrent_m4_after_m3.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('eval_experiment="${eval_base_experiment}-recovery${recovery_wave}"', text)

    def test_original_m4_stands_down_when_causal_rescue_owns_the_pilot(self) -> None:
        text = (ROOT / "scripts" / "resume_recurrent_m4_after_m3.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('causal_rescue_marker="$state_dir/causal_rescue_in_progress"', text)
        self.assertGreaterEqual(text.count('if [[ -f "$causal_rescue_marker" ]]'), 2)
        self.assertIn('touch "$state_dir/original_pilot_superseded_by_causal_rescue"', text)
        self.assertIn('wait_pass_recoverable "$eval_experiment" 5 "M4 formal evaluation"', text)
        self.assertIn("M4 formal evaluation exhausted three recovery waves", text)

    def test_checkpoint_probe_recovers_terminal_failures(self) -> None:
        text = (ROOT / "scripts" / "resume_checkpoint250_c4_probe.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("for recovery_wave in 0 1 2 3", text)
        self.assertIn('experiment="${base_experiment}-recovery${recovery_wave}"', text)
        self.assertIn("C4 probe attempt failed", text)
        self.assertIn("C4 probe exhausted three recovery waves", text)

    def test_local_checkpoint_curve_waits_for_atomic_artifacts(self) -> None:
        text = (ROOT / "scripts" / "watch_m4_checkpoint_c4_curve_local.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("rounds=(500 1000 2000 3000)", text)
        self.assertIn("adapter/adapter_model.safetensors", text)
        self.assertIn("checkpoint_completion.json", text)
        self.assertIn("receipt['sha256'][key]", text)
        self.assertIn("wait_for_gpu", text)
        self.assertIn("local C4 exhausted three attempts", text)

    def test_remote_checkpoint_curve_closes_when_original_m4_is_superseded(self) -> None:
        text = (ROOT / "scripts" / "watch_m4_checkpoint_c4_curve_feeds.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("original_pilot_superseded_by_causal_rescue", text)
        self.assertIn("superseded_after_round1750", text)
        self.assertIn("refresh_curve", text)

    def test_causal_strength_monitor_recovers_failed_branches(self) -> None:
        text = (ROOT / "scripts" / "watch_m4_causal_strength_branches.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("for recovery_wave in 0 1 2 3", text)
        self.assertIn("terminal_nonpass_job_names", text)
        self.assertIn("Causal-strength branches exhausted three recovery waves", text)
        self.assertIn("summarize_m4_causal_strength_branches.py", text)
        self.assertIn('download_passed "$active_experiment"', text)
        self.assertIn("no passed siblings to download", text)

    def test_causal_escalation_accumulates_passed_siblings_across_recovery(self) -> None:
        text = (ROOT / "scripts" / "watch_m4_causal_strength_escalation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("terminal_nonpass_job_names", text)
        self.assertIn('download_passed "$active_experiment"', text)
        self.assertIn("no passed siblings to download", text)
        self.assertIn("--expected-branches raw8 mean2_normalized", text)

    def test_causal_escalation_validator_accepts_amlt_banner_only_before_yaml(self) -> None:
        import tempfile

        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_strength_escalation import load as load_base, render
        from validate_m4_causal_strength_escalation_amlt_config import load, validate

        config = render(
            load_base(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"),
            ["raw8", "mean2_normalized"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dump.yaml"
            path.write_text(
                "Amulet version banner that is not YAML\n" + yaml.safe_dump(config),
                encoding="utf-8",
            )
            validate(load(path))

    def test_causal_rescue_validator_accepts_amlt_banner_only_before_yaml(self) -> None:
        import tempfile

        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import load as load_base, render_full
        from validate_m4_causal_rescue_amlt_config import load, validate

        config = render_full(
            load_base(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"),
            ["raw4"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dump.yaml"
            path.write_text(
                "Amulet version banner that is not YAML\n" + yaml.safe_dump(config),
                encoding="utf-8",
            )
            validate(load(path))

    def test_causal_rescue_requires_full_and_formal_gates_before_promotion(self) -> None:
        text = (ROOT / "scripts" / "watch_m4_causal_rescue_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("validate_m4_causal_rescue_result.py", text)
        self.assertIn("validate_m4_pilot_result.py", text)
        self.assertIn("for recovery_wave in 0 1 2 3", text)
        self.assertLess(
            text.index("validate_m4_pilot_result.py"),
            text.index('az storage blob upload --account-name "$account"'),
        )
        self.assertLess(
            text.index('cmp -s "$state_dir/selected_receipt.json"'),
            text.index('touch "$m4_dir/pilot_complete"'),
        )
        self.assertGreaterEqual(text.count("stand_down_if_original_passed"), 3)


if __name__ == "__main__":
    unittest.main()
