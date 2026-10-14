from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WatchState8UnifiedFinalFitTest(unittest.TestCase):
    def test_full_fit_waits_for_scientific_gate_and_recovers(self) -> None:
        text = (ROOT / "scripts/watch_state8_unified_final_fit.sh").read_text()
        for fragment in (
            "state8_semantic_candidate_selection_v1",
            "select_state8_semantic_candidates.py",
            "eedi_g050",
            "publish_receipt.json",
            "selected_adapter_sha",
            "selected_state_sha",
            "initial_artifact_sha256",
            '":$job"',
            "scientific_gate_pass",
            "no semantic T3+C4 candidate passed",
            "required_units':16",
            "validate_state8_unified_final_fit_amlt.py",
            "wave=$((wave+1))",
            "MAX_RECOVERY_WAVES:-32",
            "259845-row unified epoch",
            "state8_final_fit_receipt.json",
            "[64,8,5120]",
            "state8_clean_prompt_teacher_provenance_gate_v2",
            "legacy_answer_adjacent_contracts",
            "state8 full-fit receipt validation failed",
            'base_experiment="${job}-20260806"',
            "has_forward_progress_job",
            "paused_job_names",
            "amlt resume",
            "sys.stdout.write(''.join(value+'\\n' for value in values))",
            "monitor_layerwise_unified_training_progress.py",
            "--total-rounds 16241",
            "--max-run-seconds 345600",
            "stalled.json",
            "health_alert.json",
        ):
            self.assertIn(fragment, text)
        self.assertNotIn("amlt cancel", text)


if __name__ == "__main__":
    unittest.main()
