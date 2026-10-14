from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class M4CausalRescuePipelineTest(unittest.TestCase):
    def test_rendered_full_configs_are_fail_closed(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import TRAIN_SHA256, load, render_full
        from validate_m4_causal_rescue_amlt_config import validate

        base = load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml")
        for branch in ("raw4", "mean_normalized"):
            rendered = render_full(base, [branch])
            validate(rendered)
            command = rendered["jobs"][0]["command"]
            self.assertIn(TRAIN_SHA256, command)
            self.assertNotIn(
                "24f066362c951edb932caf2b98c4cf909b7a211cc090a7a3623946d659bfc1d2",
                command,
            )
            self.assertIn("checkpoint-round-500", command)
            self.assertNotIn("--diagnostic-stop-after-round", command)
            self.assertIn(
                '--resume-from-checkpoint "$$resume_checkpoint"', command
            )
            self.assertIn("mapfile -t rescue_checkpoints", command)
            self.assertNotIn("RESUME_CHECKPOINT", command)
            self.assertIn("causal_rescue/selected_seed42.json", command)
            self.assertIn("final_fit_causal_repeat_fraction", command)
            self.assertIn("candidate_rejected", command)
            self.assertNotIn(
                "assert summary.get('scientific_gate_pass') is True", command
            )

    def test_dense_rehearsal_can_be_continued_without_protocol_drift(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import load, render_full
        from validate_m4_causal_rescue_amlt_config import validate

        base = load(ROOT / "amlt/feeds_m4_causal_rehearsal_20260731.yaml")
        for branch in ("repeat100_w1", "repeat100_w4"):
            rendered = render_full(base, [branch])
            validate(rendered)
            command = rendered["jobs"][0]["command"]
            self.assertIn("checkpoint-round-500", command)
            self.assertIn("--causal-repeat-fraction 1.0", command)
            self.assertIn("completion.get('causal_pairs_seen') == 12000", command)
            self.assertIn("completion.get('unique_causal_pairs_seen') == 1726", command)
            self.assertIn("--resume-from-checkpoint \"$$resume_checkpoint\"", command)
            self.assertNotIn("--diagnostic-stop-after-round", command)
            self.assertIn("candidate_rejected", command)

    def test_bootstrap_rehearsal_can_be_promoted_to_symmetric_full_fit(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import TRAIN_SHA256, load, render_full
        from validate_m4_causal_rescue_amlt_config import validate

        base = load(ROOT / "amlt/feeds_m4_causal_bootstrap_rehearsal_20260731.yaml")
        for branch in ("bootstrap_sum_w1", "bootstrap_sum_w4"):
            rendered = render_full(base, [branch])
            validate(rendered)
            command = rendered["jobs"][0]["command"]
            self.assertEqual(command.count(TRAIN_SHA256), 1)
            self.assertIn("checkpoint-round-500", command)
            self.assertIn("--causal-preference-reduction sum", command)
            self.assertIn("--causal-repeat-fraction 1.0", command)
            self.assertNotIn("--causal-objective-mode bootstrap_then_symmetric", command)
            self.assertNotIn("--diagnostic-stop-after-round", command)

        from render_m4_causal_bootstrap_rehearsal import render as render_bootstrap

        native_base = render_bootstrap(
            load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"),
            ["bootstrap_native25_sum_w1", "bootstrap_native25_sum_w4"],
        )
        for branch in ("bootstrap_native25_sum_w1", "bootstrap_native25_sum_w4"):
            rendered = render_full(native_base, [branch])
            validate(rendered)
            command = rendered["jobs"][0]["command"]
            self.assertIn("checkpoint-round-500", command)
            self.assertIn("--causal-preference-reduction sum", command)
            self.assertNotIn("--causal-objective-mode bootstrap_then_symmetric", command)

        balanced_base = render_bootstrap(
            load(ROOT / "amlt/feeds_m4_causal_rehearsal_sum_20260731.yaml"),
            ["bootstrap_balanced_sum_w1", "bootstrap_balanced_sum_w4"],
        )
        rendered = render_full(balanced_base, ["bootstrap_balanced_sum_w1"])
        validate(rendered)
        command = rendered["jobs"][0]["command"]
        self.assertIn("--causal-objective-mode directional", command)
        self.assertIn("causal_objective_mode') == 'bootstrap_then_directional'", command)
        self.assertIn("'final_fit_causal_objective_mode': 'directional'", command)

    def test_cf4_diagnostic_can_continue_without_reverting_to_single_query_data(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import load, render_full
        from validate_m4_causal_rescue_amlt_config import validate

        base = load(ROOT / "amlt/feeds_m4_cf4_diagnostic_20260731.yaml")
        for branch in (
            "cf4_bootstrap_symmetric_w1",
            "cf4_donor_directional_w1",
        ):
            rendered = render_full(base, [branch])
            validate(rendered)
            command = rendered["jobs"][0]["command"]
            self.assertIn("checkpoint-round-250", command)
            self.assertIn("unique_causal_pairs_seen') == 1051", command)
            self.assertIn("cf4_balanced_paired_state_swap_train", command)
            self.assertIn("'causal_data_mode': 'cf4_balanced'", command)
            self.assertNotIn("--diagnostic-stop-after-round", command)

    def test_eval_uses_isolated_candidate_receipt(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import load, render_eval
        from validate_m4_eval_amlt_config import validate

        base = load(ROOT / "amlt/feeds_recurrent_student_m4_eval_20260730.yaml")
        rendered = render_eval(base)
        validate(rendered)
        for job in rendered["jobs"]:
            self.assertIn("m4/causal_rescue/selected_seed42.json", job["command"])

    def test_escalation_configs_are_controlled_and_validated(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_strength_escalation import load, render
        from render_m4_causal_strength_escalation import LEGACY_TRAIN_SHA256, TRAIN_SHA256
        from validate_m4_causal_strength_escalation_amlt_config import validate

        base = load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml")
        rendered = render(base, ["raw8", "mean2_normalized"])
        validate(rendered)
        envs = {
            job["submit_args"]["env"]["BRANCH"]: job["submit_args"]["env"]
            for job in rendered["jobs"]
        }
        self.assertEqual(envs["raw8"]["CAUSAL_WEIGHT"], "8.0")
        self.assertEqual(envs["mean2_normalized"]["CAUSAL_WEIGHT"], "2.0")
        self.assertEqual(envs["mean2_normalized"]["NORMALIZE_CAUSAL"], "1")
        for job in rendered["jobs"]:
            command = job["command"]
            self.assertIn(TRAIN_SHA256, command)
            self.assertNotIn(LEGACY_TRAIN_SHA256, command)
            self.assertIn(
                "assert branch in {'raw8', 'mean2_normalized'}, branch", command
            )

    def test_escalated_current_hash_branches_render_full_rescue(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        from render_m4_causal_rescue import TRAIN_SHA256, render_full
        from render_m4_causal_strength_escalation import load, render
        from validate_m4_causal_rescue_amlt_config import validate

        escalated = render(
            load(ROOT / "amlt/feeds_m4_causal_strength_branches_20260731.yaml"),
            ["raw8", "mean2_normalized"],
        )
        for branch in ("raw8", "mean2_normalized"):
            rendered = render_full(escalated, [branch])
            validate(rendered)
            command = rendered["jobs"][0]["command"]
            self.assertEqual(command.count(TRAIN_SHA256), 1)
            self.assertNotIn("--diagnostic-stop-after-round", command)

    def test_candidate_selector_prefers_best_eligible_gate(self) -> None:
        import json
        import subprocess

        sparse = {
            "stage": "M4_causal_strength_round500",
            "branches": [
                {
                    "branch": "mean_normalized",
                    "C4_D_specific": 0.18,
                    "C4_clean_n": 55,
                    "C4_noop_flip": 0.0,
                    "C4_same_rule_flip": 0.05,
                    "validation_accuracy_gap": -0.01,
                }
            ],
        }
        dense = {
            "stage": "M4_causal_strength_round500",
            "branches": [
                {
                    "branch": "repeat100_w1",
                    "C4_D_specific": 0.24,
                    "C4_clean_n": 56,
                    "C4_noop_flip": 0.0,
                    "C4_same_rule_flip": 0.04,
                    "validation_accuracy_gap": -0.02,
                },
                {
                    "branch": "repeat100_w4",
                    "C4_D_specific": 0.26,
                    "C4_clean_n": 56,
                    "C4_noop_flip": 0.0,
                    "C4_same_rule_flip": 0.16,
                    "validation_accuracy_gap": -0.01,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sparse.json").write_text(json.dumps(sparse), encoding="utf-8")
            (root / "dense.json").write_text(json.dumps(dense), encoding="utf-8")
            output = root / "choice.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/select_m4_causal_candidate.py"),
                    "--candidate",
                    f"sparse={root / 'sparse.json'}",
                    "--candidate",
                    f"dense={root / 'dense.json'}",
                    "--output",
                    str(output),
                ],
                check=True,
            )
            choice = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(choice["scientific_gate_pass"])
        self.assertEqual(choice["selected_branch"], "repeat100_w1")
        self.assertEqual(choice["selected_family"], "dense")
        self.assertEqual(
            [row["branch"] for row in choice["ranked_candidates"]],
            ["repeat100_w1", "mean_normalized"],
        )

    def test_rescue_controller_falls_back_across_scientific_candidates(self) -> None:
        rescue = (ROOT / "scripts/watch_m4_causal_rescue_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ranked_candidates", rescue)
        self.assertIn("for candidate in", rescue)
        self.assertIn("candidate scientific gate failed", rescue)
        self.assertIn("selected_candidate=true", rescue)
        self.assertIn("all eligible causal-rescue candidates failed", rescue)

    def test_final_controller_waits_for_active_rescue(self) -> None:
        text = (ROOT / "scripts/resume_recurrent_final_after_m4.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("causal_rescue_in_progress", text)
        self.assertIn("causal-rescue controller is active", text)
        self.assertIn(
            '[[ ! -f "$m4_dir/pilot_complete" || -f "$causal_rescue_marker" ]]',
            text,
        )
        self.assertIn("waiting for causal-rescue receipt ownership to settle", text)

    def test_rescue_controller_waits_for_persistent_donor_directional_family(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_rescue_pipeline.sh").read_text()
        for required in (
            "causal_donor_directional",
            "donor_directional_sum_w1",
            "donor_directional_sum_w4",
            "donor_directional=$donor_directional_dir/comparison.json",
            "feeds_m4_causal_donor_directional_20260731.yaml",
        ):
            self.assertIn(required, text)
        rescue = (ROOT / "scripts/watch_m4_causal_rescue_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("watch_m4_causal_strength_escalation.sh", rescue)
        self.assertIn('escalation_dir/controller.active.pid', rescue)
        self.assertIn('kill -0 "$(cat "$escalation_dir/controller.active.pid")"', rescue)
        self.assertIn('while [[ ! -f "$escalation_dir/complete" ]]', rescue)
        self.assertIn("causal_rehearsal/complete", rescue)
        self.assertIn("select_m4_causal_candidate.py", rescue)
        self.assertIn("stand_down_if_original_passed", rescue)
        self.assertIn("standing down", rescue)
        self.assertIn(
            '[[ -f "$branch_dir/complete" || ! -f "$branch_dir/needs_attention" ]]',
            rescue,
        )
        self.assertIn('sum_rehearsal_early_stopped', rescue)
        self.assertIn("causal_bootstrap_rehearsal", rescue)
        self.assertIn("causal_bootstrap_directional_tail", rescue)
        self.assertIn("bootstrap_directional", rescue)
        self.assertIn(
            "directional-tail rehearsal unavailable; selecting from completed families",
            rescue,
        )
        self.assertIn("allowed_families = {", rescue)
        self.assertIn("'bootstrap_directional', 'escalation'", rescue)
        self.assertIn('candidate_args+=', rescue)
        self.assertNotIn('$${candidate_args[@]}', rescue)
        self.assertIn('bootstrap causal rehearsal unavailable', rescue)
        self.assertIn('bootstrap_rehearsal_early_stopped', rescue)
        self.assertIn('excluding cancelled-gradient objective', rescue)
        self.assertIn('feeds_m4_causal_bootstrap_rehearsal_selected_20260731.yaml', rescue)
        self.assertIn('feeds_m4_causal_bootstrap_directional_tail_20260731.yaml', rescue)
        self.assertIn('rehearsal_early_stopped', rescue)
        self.assertIn('excluding dead objective', rescue)
        self.assertIn('sum_rehearsal_early_stopped', rescue)
        self.assertIn('bootstrap supersedes it', rescue)


if __name__ == "__main__":
    unittest.main()
