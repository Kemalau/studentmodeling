from __future__ import annotations

import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


class WatchLayerwiseUnifiedFinalFitTest(unittest.TestCase):
    def test_controller_orders_scientific_selection_before_submission(self):
        text=(ROOT/'scripts/watch_layerwise_unified_final_fit.sh').read_text()
        self.assertIn('[[ ! -f "$pilot_dir/scientific_gate_pass" ]]',text)
        self.assertIn("selected_C4",text)
        self.assertIn('wait_quota 16 "$state_dir/quota_before_submission.json"',text)
        self.assertIn("'required_units':n",text)
        self.assertIn("validate_layerwise_unified_final_fit_amlt.py",text)
        self.assertIn("all_expected_jobs_pass(sys.argv[1],1)",text)
        self.assertIn("real_examples']==259845",text)
        self.assertIn("final_fit/seed42.json",text)
        self.assertIn('active_experiment.txt',text)
        self.assertIn('[[ -s "$active_experiment_file" ]]',text)
        self.assertIn('max_resilient_recoveries="${MAX_RESILIENT_RECOVERIES:-12}"',text)

    def test_controller_restart_reuses_verified_selection_without_auth_failure(self):
        text=(ROOT/'scripts/watch_layerwise_unified_final_fit.sh').read_text()
        self.assertIn('if ! cmp -s "$pilot_dir/selection.json" "$state_dir/selection_verification.json"',text)
        self.assertIn("temporary monitoring-token expiry",text)
        self.assertIn("while ! az account get-access-token",text)
        self.assertNotIn('|| fail "could not select fixed subscription"',text)

    def test_controller_recovers_a_real_training_stall_but_spares_validation(self):
        text=(ROOT/'scripts/watch_layerwise_unified_final_fit.sh').read_text()
        self.assertIn('timeout 90 amlt status "$experiment"',text)
        self.assertIn('timeout 90 amlt show "$experiment"',text)
        self.assertIn("seconds_since_progress', 0)) >= 7200",text)
        self.assertIn("int(x.get('round', 0)) < 16000",text)
        self.assertIn("<= 900",text)
        self.assertIn("x.get('observed_at')",text)
        self.assertIn('amlt cancel "$experiment" --status running -y',text)
        self.assertIn('atomic-checkpoint recovery',text)
        self.assertLess(text.index('stalled.cancel.txt'),text.index('resilient recovery dump failed'))

    def test_controller_hashes_downloaded_model_artifacts(self):
        text=(ROOT/'scripts/watch_layerwise_unified_final_fit.sh').read_text()
        self.assertIn('artifact_manifest.json',text)
        self.assertIn("'algorithm': 'sha256'",text)
        self.assertIn("'adapter_model.safetensors', 'recurrent_state.pt'",text)

    def test_controller_prefers_single_gpu_finalization_after_exact_final_checkpoint(self):
        text=(ROOT/'scripts/watch_layerwise_unified_final_fit.sh').read_text()
        self.assertIn('checkpoint-round-16241',text)
        self.assertIn("x.get('round')==16241",text)
        self.assertIn("x.get('world_size')==16",text)
        self.assertIn('wait_quota 1',text)
        self.assertIn('validate_layerwise_unified_final_validation_recovery_amlt.py',text)
        self.assertIn('single-A100 exact-final-checkpoint finalizer',text)
        self.assertLess(text.index('if final_checkpoint_ready; then'),text.index('if (( recovery_wave >= max_resilient_recoveries ))'))
        self.assertIn('finalizer exhausted three retries',text)

    def test_controller_preserves_early_checkpoint_before_pruning_recovery(self):
        text=(ROOT/'scripts/watch_layerwise_unified_final_fit.sh').read_text()
        self.assertIn('checkpoint-round-1000',text)
        self.assertIn('early_snapshot_dir=',text)
        self.assertIn('early_snapshot_dir/snapshot_complete',text)
        self.assertIn('early_snapshot_dir/needs_attention',text)
        self.assertLess(text.index('waiting for immutable round-1000 snapshot'),text.index('recovery_wave=$((recovery_wave + 1))'))


if __name__=='__main__': unittest.main()
