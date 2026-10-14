from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_promoted_checkpoint_trajectory import EXPECTED_SUPPORT, summarize


def test_joint_trajectory_screen_and_round1000_deltas() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        groups = {}
        for name, support in EXPECTED_SUPPORT.items():
            task = name.split("/", 1)[0]
            row = {"n": support}
            if task == "predict_correctness":
                row["macro_f1"] = 0.6
            elif task == "predict_error_persistence":
                row["macro_f1"] = 0.4
            else:
                row.update(format_valid_rate=1.0, normalized_exact_match=0.3)
            groups[name] = row
        metrics = {
            "state_architecture": "layerwise",
            "state_shape": [64, 5120],
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "sampled_rows": sum(EXPECTED_SUPPORT.values()),
            "metrics_by_group": groups,
        }
        baseline = json.loads(json.dumps(metrics))
        for name, row in baseline["metrics_by_group"].items():
            if name.startswith("diagnose_misconception/"):
                row["format_valid_rate"] = 0.7
        semantic = {
            "checkpoint_round": 1250,
            "support": 75,
            "semantic_accuracy": 0.5,
            "reference_semantic_accuracy": 0.4,
            "retention_ratio": 1.25,
            "minimum_retention_ratio": 0.85,
            "pass": True,
            "judge_backend": "trapi",
            "judge_rubric_sha256": "a" * 64,
        }
        baseline_semantic = {"semantic_accuracy": 0.25, "retention_ratio": 0.6, "pass": False}
        c4 = {
            "summary": {
                "all_pairs_n": 50,
                "clean_n": 30,
                "D_specific": 0.2,
                "scientific_gate_pass": True,
                "checks": {"specificity": True},
                "clean": {
                    "swap": {"flip_to_A_rate": 0.4},
                    "random": {"flip_to_A_rate": 0.2},
                    "samemal": {"flip_to_A_rate": 0.05},
                    "noop": {"flip_to_A_rate": 0.0},
                },
            }
        }
        for name, value in (
            ("metrics.json", metrics),
            ("baseline.json", baseline),
            ("semantic.json", semantic),
            ("baseline_semantic.json", baseline_semantic),
            ("c4.json", c4),
        ):
            (root / name).write_text(json.dumps(value))
        value = summarize(
            root / "metrics.json",
            root / "semantic.json",
            root / "c4.json",
            checkpoint_round=1250,
            baseline_metrics=root / "baseline.json",
            baseline_semantic=root / "baseline_semantic.json",
        )
        assert value["joint_diagnostic_pass"]
        assert abs(value["task_delta_from_round1000"]["T3_format_valid_rate"] - 0.3) < 1e-12
        assert value["semantic_accuracy_delta_from_round1000"] == 0.25


def test_missing_support_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        groups = {}
        for name, support in EXPECTED_SUPPORT.items():
            task = name.split("/", 1)[0]
            row = {"n": support}
            if task == "predict_correctness":
                row["macro_f1"] = 0.6
            elif task == "predict_error_persistence":
                row["macro_f1"] = 0.4
            else:
                row.update(format_valid_rate=1.0, normalized_exact_match=0.3)
            groups[name] = row
        groups.pop(next(iter(groups)))
        metrics = {
            "state_architecture": "layerwise",
            "state_shape": [64, 5120],
            "raw_history_visible_to_decoder": False,
            "persistent_history_kv_cache": False,
            "sampled_rows": sum(row["n"] for row in groups.values()),
            "metrics_by_group": groups,
        }
        semantic = {
            "checkpoint_round": 1250,
            "semantic_accuracy": 0.5,
            "pass": True,
        }
        c4 = {"summary": {"scientific_gate_pass": True, "clean": {}}}
        for name, value in (("metrics.json", metrics), ("semantic.json", semantic), ("c4.json", c4)):
            (root / name).write_text(json.dumps(value))
        value = summarize(
            root / "metrics.json",
            root / "semantic.json",
            root / "c4.json",
            checkpoint_round=1250,
        )
        assert not value["joint_diagnostic_pass"]
        assert not value["checks"]["exact_protocol_support"]
