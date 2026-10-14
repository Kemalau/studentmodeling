from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PairOverfitPipelineTest(unittest.TestCase):
    def test_watcher_is_recoverable_and_fail_closed(self) -> None:
        text = (ROOT / "scripts/watch_m4_causal_pair_overfit.sh").read_text(encoding="utf-8")
        for value in (
            "feeds_submission_global.lock",
            "validate_m4_causal_pair_overfit_amlt_config.py",
            "classify_experiment_status",
            "terminal_nonpass_job_names",
            "summarize_m4_causal_pair_overfit.py",
            'touch "$state_dir/complete"',
            "timeout 180s amlt logs view",
        ):
            self.assertIn(value, text)

    def test_summary_detects_trainable_path_and_sum_advantage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for branch, reduction, first, last, flip in (
                ("pair8_mean", "mean", 0.50, 0.48, 0.1),
                ("pair8_sum", "sum", 0.50, 0.10, 0.8),
            ):
                path = root / branch / "metrics/pair_overfit_summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "branch": branch,
                    "preference_reduction": reduction,
                    "margin_loss_first_50": first,
                    "margin_loss_last_50": last,
                    "margin_loss_last": last,
                    "causal_validation": {
                        "clean_n": 8,
                        "directed_flip_to_A_on_clean": flip,
                    },
                }))
            output = root / "comparison.json"
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/summarize_m4_causal_pair_overfit.py"),
                "--root", str(root), "--output", str(output),
            ], check=True)
            result = json.loads(output.read_text())
        self.assertTrue(result["path_trainable"])
        self.assertTrue(result["sum_alignment_supported"])


if __name__ == "__main__":
    unittest.main()
