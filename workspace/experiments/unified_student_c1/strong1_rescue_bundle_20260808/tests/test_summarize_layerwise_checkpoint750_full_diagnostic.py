from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_layerwise_checkpoint750_full_diagnostic import summarize  # noqa: E402
from summarize_layerwise_unified_final_eval import FROZEN_GROUP_SUPPORT  # noqa: E402


METRICS = {
    "predict_correctness": {"macro_f1": 0.70},
    "predict_response": {"math_equivalence": 0.55},
    "diagnose_misconception": {
        "format_valid_rate": 1.0,
        "normalized_exact_match": 0.1,
    },
    "predict_prospective_error": {"joint_math_equivalence": 0.80},
    "predict_error_persistence": {"macro_f1": 0.60},
}


def write_metrics(path: Path, tasks: set[str]) -> None:
    groups = {}
    for name, support in FROZEN_GROUP_SUPPORT.items():
        task = name.split("/", 1)[0]
        if task in tasks:
            groups[name] = {"n": support, **METRICS[task]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "state_architecture": "layerwise",
        "state_shape": [64, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "sampled_rows": sum(int(row["n"]) for row in groups.values()),
        "metrics_by_group": groups,
    }))


def fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    formal = root / "formal"
    write_metrics(formal / "t1/metrics.json", {"predict_correctness"})
    write_metrics(formal / "t2/metrics.json", {"predict_response"})
    write_metrics(formal / "t3/metrics.json", {"diagnose_misconception"})
    write_metrics(formal / "t5/metrics.json", {"predict_prospective_error"})
    t6 = root / "t6/metrics.json"
    write_metrics(t6, {"predict_error_persistence"})
    c4 = root / "c4/recurrent_causal_gate.json"
    c4.parent.mkdir()
    c4.write_text(json.dumps({
        "state_architecture": "layerwise",
        "state_shape": [64, 5120],
        "raw_history_visible_to_decoder": False,
        "persistent_history_kv_cache": False,
        "summary": {
            "clean_n": 80,
            "D_specific": 0.20,
            "clean": {
                "swap": {"flip_to_A_rate": 0.30},
                "random": {"flip_to_A_rate": 0.10},
                "samemal": {"flip_to_A_rate": 0.05},
                "noop": {"flip_to_A_rate": 0.0},
            },
        },
    }))
    semantic = root / "semantic/metrics.json"
    semantic.parent.mkdir()
    semantic.write_text(json.dumps({
        "judge_backend": "trapi",
        "judge_effective_model": "judge",
        "metrics_by_group": {
            "g": {"llm_judge_n": 10, "llm_judge_accuracy": 0.4},
        },
    }))
    return formal, t6, c4, semantic


class Checkpoint750FullSummaryTest(unittest.TestCase):
    def test_complete_closed_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            value = summarize(*args)
            self.assertTrue(value["diagnostic_gate_pass"])
            self.assertEqual(value["sampled_rows"], 21676)
            self.assertEqual(value["T3"]["semantic"]["llm_judge_accuracy"], 0.4)

    def test_history_leak_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal, t6, c4, semantic = fixture(Path(temporary))
            path = formal / "t1/metrics.json"
            value = json.loads(path.read_text())
            value["raw_history_visible_to_decoder"] = True
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Raw history leak"):
                summarize(formal, t6, c4, semantic)

    def test_scientific_c4_failure_is_reported_without_parse_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal, t6, c4, semantic = fixture(Path(temporary))
            value = json.loads(c4.read_text())
            value["summary"]["D_specific"] = 0.12
            c4.write_text(json.dumps(value))
            summary = summarize(formal, t6, c4, semantic)
            self.assertFalse(summary["diagnostic_gate_pass"])
            self.assertFalse(summary["checks"]["C4_specificity"])


if __name__ == "__main__":
    unittest.main()
