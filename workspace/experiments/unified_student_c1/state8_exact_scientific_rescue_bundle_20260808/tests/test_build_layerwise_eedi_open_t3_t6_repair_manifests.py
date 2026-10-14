from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_layerwise_eedi_open_t3_t6_repair_manifests import build  # noqa: E402
from build_layerwise_task_retention_manifests import read_jsonl  # noqa: E402


def test_repair_manifests_are_fresh_balanced_and_exact() -> None:
    train = read_jsonl(ROOT / "data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl")
    formal = read_jsonl(ROOT / "data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl")
    import json

    prior = json.loads(
        (ROOT / "manifests/layerwise_task_retention_eedi_full_seed42_v1.json").read_text()
    )
    prior_indices = set(prior["indices"])
    expected = {
        "balanced": {"diagnose_misconception": 200, "predict_error_persistence": 600},
        "strong": {"diagnose_misconception": 100, "predict_error_persistence": 700},
    }
    for variant, task_subset in expected.items():
        value = build(train, formal, prior, variant=variant, seed=42)
        assert value["start_global_position"] == 17_000
        assert value["segment_rows"] == 1_000
        assert len(value["indices"]) == len(set(value["indices"])) == 1_000
        assert not prior_indices.intersection(value["indices"])
        assert value["formal_example_id_overlap"] == 0
        assert value["t6_label_counts"]["persistent"] == value["t6_label_counts"]["transient"]
        for task, count in task_subset.items():
            assert value["task_counts"][task] == count
