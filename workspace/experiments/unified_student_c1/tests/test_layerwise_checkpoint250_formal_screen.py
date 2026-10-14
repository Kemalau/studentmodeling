from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_layerwise_checkpoint250_formal_screen import render  # noqa: E402
from summarize_layerwise_checkpoint250_formal_screen import (  # noqa: E402
    EXPECTED_SUPPORT,
    summarize,
)
from validate_layerwise_checkpoint250_formal_screen_amlt import validate  # noqa: E402


class CheckpointFormalScreenTest(unittest.TestCase):
    def test_watcher_is_bound_to_the_rendered_round(self) -> None:
        text = (ROOT / "scripts/watch_layerwise_checkpoint250_formal_screen.sh").read_text()
        self.assertIn('[[ "$checkpoint_round" == 1000 ]]', text)
        self.assertIn('Round-${checkpoint_round} five-task diagnostic recovery=', text)
        self.assertIn('[[ -f "$auth_required" ]]', text)
        self.assertIn("while ! az account get-access-token", text)
        self.assertIn("attention_is_current", text)
        self.assertIn('active_experiment.txt" -nt "$marker"', text)
        self.assertIn('"$active_pid" -nt "$marker"', text)
        self.assertNotIn('Round-250 five-task diagnostic recovery=', text)

    def test_rendered_source_config_is_valid_and_immutable(self) -> None:
        base = yaml.safe_load(
            (ROOT / "amlt/feeds_layerwise_unified_final_eval_20260731.yaml").read_text()
        )
        value = render(base)
        validate(value)
        self.assertEqual(len(value["jobs"]), 4)
        for job in value["jobs"]:
            command = job["command"]
            self.assertIn("checkpoint-round-1000", command)
            self.assertIn("adapter/adapter_config.json", command)
            self.assertIn("--max-samples-per-group 25", command)
            self.assertIn("scripts/frozen_layerwise_finalfit/layerwise_recurrent_student_model.py", command)
            self.assertIn("tests/frozen_layerwise_finalfit/test_layerwise_recurrent_student_model.py", command)
            self.assertNotIn("\n+        --max-length", command)
            self.assertNotIn("final_fit/seed42.json", command)
            self.assertNotIn("final_adapter", command)

    def test_validator_rejects_literal_patch_marker_argument(self) -> None:
        base = yaml.safe_load(
            (ROOT / "amlt/feeds_layerwise_unified_final_eval_20260731.yaml").read_text()
        )
        value = render(base)
        value["jobs"][0]["command"] = value["jobs"][0]["command"].replace(
            "\n        --max-length", "\n+        --max-length"
        )
        with self.assertRaisesRegex(ValueError, "Patch marker"):
            validate(value)

    def test_summary_requires_all_groups_and_catastrophic_floors(self) -> None:
        by_job: dict[str, dict[str, dict]] = {}
        for name, support in EXPECTED_SUPPORT.items():
            task = name.split("/", 1)[0]
            row = {"n": support}
            if task in {"predict_correctness", "predict_error_persistence"}:
                row["macro_f1"] = 0.8
            elif task == "predict_response":
                row["math_equivalence"] = 0.8
                row["format_valid_rate"] = 1.0
            elif task == "predict_prospective_error":
                row["joint_math_equivalence"] = 0.8
                row["format_valid_rate"] = 1.0
            else:
                row["format_valid_rate"] = 1.0
            job = (
                "t1_t6"
                if task in {"predict_correctness", "predict_error_persistence"}
                else task
            )
            by_job.setdefault(job, {})[name] = row
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, groups in enumerate(by_job.values()):
                job = root / f"job-{index}"
                job.mkdir()
                (job / "metrics.json").write_text(
                    json.dumps(
                        {
                            "state_architecture": "layerwise",
                            "state_shape": [64, 5120],
                            "raw_history_visible_to_decoder": False,
                            "persistent_history_kv_cache": False,
                            "sampled_rows": sum(int(row["n"]) for row in groups.values()),
                            "metrics_by_group": groups,
                        }
                    )
                )
            value = summarize(root)
            self.assertTrue(value["diagnostic_gate_pass"])
            self.assertEqual(value["checkpoint_round"], 1000)
            checkpoint750 = summarize(root, checkpoint_round=750)
            self.assertEqual(checkpoint750["checkpoint_round"], 750)
            self.assertEqual(
                checkpoint750["stage"],
                "layerwise_finalfit_checkpoint750_formal_screen",
            )
            metric = next(root.rglob("metrics.json"))
            broken = json.loads(metric.read_text())
            broken["metrics_by_group"].pop(next(iter(broken["metrics_by_group"])))
            metric.write_text(json.dumps(broken))
            self.assertFalse(summarize(root)["checks"]["exact_protocol_support"])

    def test_summary_accepts_one_file_with_exact_full_protocol_support(self) -> None:
        groups = {}
        for name, support in EXPECTED_SUPPORT.items():
            task = name.split("/", 1)[0]
            row = {"n": support}
            if task in {"predict_correctness", "predict_error_persistence"}:
                row["macro_f1"] = 0.8
            elif task == "predict_response":
                row["math_equivalence"] = 0.8
            elif task == "predict_prospective_error":
                row["joint_math_equivalence"] = 0.8
            else:
                row["format_valid_rate"] = 1.0
            groups[name] = row
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metrics.json").write_text(json.dumps({
                "state_architecture": "layerwise",
                "state_shape": [64, 5120],
                "raw_history_visible_to_decoder": False,
                "persistent_history_kv_cache": False,
                "sampled_rows": sum(EXPECTED_SUPPORT.values()),
                "metrics_by_group": groups,
            }))
            self.assertTrue(summarize(root, checkpoint_round=750)["diagnostic_gate_pass"])

    def test_summary_rejects_unsupported_shard_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(3):
                job = root / f"job-{index}"
                job.mkdir()
                (job / "metrics.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "one, two or four"):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
