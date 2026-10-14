from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RecurrentUnifiedSupervisorTest(unittest.TestCase):
    def test_supervisor_exposes_feeds_auth_state(self) -> None:
        text = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text()
        self.assertIn("az account get-access-token", text)
        self.assertIn("feeds_auth\\tready", text)
        self.assertIn("feeds_auth\\tauth_required", text)
        self.assertIn("feeds_reauth\\tpending", text)
        self.assertIn("watch_layerwise_unified_reauth.sh", text)
        self.assertIn('mkdir -p "$state_dir/reauth"', text)
        self.assertIn("$project_dir/results/recurrent_unified_supervisor_20260731/reauth/controller.tmux.log", text)
        self.assertNotIn(
            "watch_layerwise_unified_reauth.sh 2>&1 | tee", text
        )
        self.assertIn('state_dir/auth_required', text)
        self.assertIn("Use the caller/default Azure CLI cache", text)
        self.assertIn("unset AZURE_CONFIG_DIR", text)
        self.assertIn("Trust an inherited Azure cache only after it proves", text)
        self.assertIn("env -u AZURE_CONFIG_DIR az account get-access-token", text)

    def test_reauth_owner_is_monitor_only_and_self_renewing(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_unified_reauth.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("reauth_layerwise_unified_pipeline.sh", text)
        self.assertIn("a fresh code will be requested", text)
        self.assertIn("az account get-access-token", text)
        self.assertIn("flock -n 9", text)
        self.assertNotIn('| tee -a "$root/auth.log"', text)
        self.assertNotIn("amlt run", text)
        self.assertNotIn("amlt cancel", text)
        self.assertIn("use_studentmodeling_feeds_auth.sh", text)
        helper = (ROOT / "scripts/use_studentmodeling_feeds_auth.sh").read_text()
        self.assertIn("unset AZURE_CONFIG_DIR", helper)
        self.assertIn(
            "env -u AZURE_CONFIG_DIR timeout 30 az account get-access-token",
            helper,
        )

    def test_supervisor_covers_every_active_pipeline_owner(self) -> None:
        text = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "watch_m4_donor_round250_c4_devicefix.sh",
            "watch_m4_donor_round500_c4_devicefix.sh",
            "watch_m4_cf4_diagnostic.sh",
            "watch_m4_cf4_single_hedge.sh",
            "watch_m2_seeded_cf4_screen.sh",
            "m2_seeded_specificity_triplet_v2",
            "EXPECTED_OBJECTIVE=specificity_triplet",
            "watch_m4_causal_rescue_pipeline.sh",
            "resume_recurrent_final_after_m4.sh",
            "CHECKPOINT_ROUND=50",
            "CHECKPOINT_ROUND=100",
            "CHECKPOINT_ROUND=400",
            "watch_layerwise_formal_screen.sh",
            "watch_layerwise_state_bridge_c4.sh",
            "watch_cross_adapter_layerwise_bridge_c4.sh",
            "watch_layerwise_recurrent_c4.sh",
            "watch_single_layer_state_bridge_c4.sh",
            "watch_layerwise_training_gate.sh",
            "watch_layerwise_recurrent_pilot.sh",
            "watch_layerwise_unified_final_fit.sh",
            "watch_layerwise_unified_training_progress.sh",
            "watch_layerwise_unified_final_eval.sh",
            "watch_layerwise_t3_semantic_judge.sh",
            "watch_layerwise_finalfit_checkpoint250_c4.sh",
            "watch_layerwise_checkpoint250_formal_screen.sh",
            "watch_layerwise_checkpoint1000_t3_semantic_judge.sh",
            "watch_layerwise_checkpoint1000_decision.sh",
            "watch_layerwise_checkpoint500_causal_rescue_recovery.sh",
            "watch_layerwise_known_oom_row_alignedmask.sh",
            "watch_layerwise_task_retention_rehearsal.sh",
            "watch_layerwise_task_retention_promoted_final_fit.sh",
            "watch_layerwise_promoted_round1000_local_c4.sh",
            "c3_exposure_matched_round1500_hashfix_v2",
            "watch_promoted_round1000_c4_n150_multiseed.sh",
            "watch_promoted_round1000_t3_semantic.sh",
            "watch_candidate_lr5e6_statepath_full_t3.sh",
            "watch_cf1_fulltask_crossadapter_full_t3.sh",
            "watch_selective_kd_lr2e6_full_t3.sh",
            "watch_current_lr2e6_fulltask_crossadapter_full_t3.sh",
            "watch_current_lr2e6_direct_teacher_t3.sh",
            "watch_layerwise_task_retention_promoted_final_eval.sh",
            "eedi_full_open_t3_selective_kd_lr_scan",
            "validate_layerwise_eedi_full_open_t3_selective_kd_lr_scan_amlt.py",
            "layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3800_local_preflight",
            "watch_layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3800_local_preflight.sh",
            "layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3950_local_c4",
            "watch_layerwise_eedi_full_open_t3_selective_kd_lr2e6_round3950_local_c4.sh",
            "watch_layerwise_eedi_open_t3_t6_repair.sh",
            "layerwise_equal_state_kd",
            "validate_layerwise_equal_state_kd_amlt.py",
            "layerwise_equal_state_kd_local_preflight",
            "layerwise_equal_state_kd_open_t3_local_preflight",
            "layerwise_equal_state_path_open_t3",
            "watch_layerwise_equal_state_path_open_t3.sh",
            "layerwise_equal_state_kd_open_t3_lr_scan",
            "watch_layerwise_equal_state_kd_open_t3_lr_scan.sh",
            "layerwise_equal_state_kd_open_t3_lr1e6",
            "watch_layerwise_equal_state_kd_open_t3_lr1e6.sh",
            "layerwise_open_t3_consolidation",
            "watch_layerwise_open_t3_consolidation.sh",
            "equal-open-t3-selective-kd",
            "watch_layerwise_checkpoint750_local_c4.sh",
            "watch_layerwise_checkpoint750_local_formal_full.sh",
            "watch_layerwise_checkpoint750_formal_full_feeds.sh",
            "watch_state8_primary_teacher_curriculum.sh",
            "watch_state8_balanced_teacher_curriculum.sh",
            "watch_state8_highkd_teacher_curriculum.sh",
            "PROMOTION_VARIANT=t3_heavy",
            "PROMOTION_VARIANT=balanced",
            "PROMOTION_VARIANT=highkd",
            "watch_state8_promotion_eval.sh",
            "watch_state8_qualified_promotion_selection.sh",
            "watch_state8_unified_final_fit.sh",
            "watch_state8_final_eval.sh",
            "watch_state8_qualified_live_status.sh",
            "watch_state8_training_progress.sh",
            "state8_training_progress_watchdog_20260805",
            "state8_qualified_complete_final_eval",
            "summarize_layerwise_unified_live_status.py",
            "summarize_state8_large_evaluation_live_status.py",
            "state8_large_evaluation_live_status_20260813/status.json",
            "live_status.json",
            '[[ -f "$root/superseded" ]]',
            '[[ -f "$root/deferred" ]]',
            "controller.active.pid",
            "needs_attention",
            "overall_complete",
            "flock -n 9",
        ):
            self.assertIn(required, text)

    def test_attention_is_not_blindly_resubmitted(self) -> None:
        text = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text(
            encoding="utf-8"
        )
        attention = text.index('[[ -f "$root/needs_attention" ]]')
        deferred = text.index('[[ -f "$root/deferred" ]]')
        restart = text.index('restart_controller "$session"')
        self.assertLess(deferred, restart)
        self.assertLess(attention, restart)

    def test_state8_watchers_use_authenticated_feeds_cache(self) -> None:
        text = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("AZURE_CONFIG_DIR=/home/v-yuejiali/.azure;", text)
        self.assertNotIn("state8_strict_stateonly_repair", text)
        self.assertIn("AZURE_CONFIG_DIR=/home/v-yuejiali/.azure-keyflip-feeds", text)

    def test_stale_downstream_attention_is_cleared_only_for_recovered_dependencies(self) -> None:
        text = (ROOT / "scripts/supervise_recurrent_unified_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("clear_stale_candidate_aggregation_attention", text)
        self.assertIn('grep \'ERROR:\' "$owner_root/controller.log"', text)
        self.assertIn('[[ ! -f "$dependency_root/needs_attention" ]]', text)
        self.assertIn('controller_alive "$dependency_root" || return 0', text)
        self.assertIn('if [[ "$last_error" == *"no task-retention branch passed every promotion gate"* ]]', text)
        self.assertIn('&& (( pending == 0 ))', text)
        self.assertIn('"promoted final fit needs attention"', text)
        self.assertIn(
            'results/layerwise_unified_final_pipeline_20260731/open_t3_consolidation',
            text,
        )

    def test_candidate_dependency_list_includes_selective_kd_in_same_call(self) -> None:
        lines = (
            ROOT / "scripts/supervise_recurrent_unified_pipeline.sh"
        ).read_text(encoding="utf-8").splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if line.strip() == "clear_stale_candidate_aggregation_attention \\"
        )
        plain = next(
            index
            for index, line in enumerate(lines[start:], start=start)
            if 'eedi_full_open_t3_lr_scan"' in line
            and "CONFIG_PATH" not in line
        )
        self.assertTrue(lines[plain].rstrip().endswith("\\"))
        self.assertIn(
            'eedi_full_open_t3_selective_kd_lr_scan"', lines[plain + 1]
        )


if __name__ == "__main__":
    unittest.main()
