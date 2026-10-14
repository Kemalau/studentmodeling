from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_layerwise_t6_calibration_manifest import build  # noqa: E402
from build_layerwise_task_retention_manifests import read_jsonl  # noqa: E402


def test_calibration_is_unused_original_validation_only() -> None:
    train = read_jsonl(ROOT / "data/processed/unified_recurrent_full259845_u4_v1/final_fit_trainvalid.jsonl")
    formal = read_jsonl(ROOT / "data/processed/recurrent_formal_evaluation_fixedu4_v2/test.jsonl")
    promotion = json.loads((ROOT / "results/layerwise_unified_final_pipeline_20260731/task_retention_promotion/eedi_full_open_t3.json").read_text())
    paths = [
        ROOT / "manifests/layerwise_task_retention_eedi_full_seed42_v1.json",
        ROOT / "manifests/layerwise_eedi_open_t3_t6_repair_balanced_seed42_v1.json",
        ROOT / "manifests/layerwise_eedi_open_t3_t6_repair_strong_seed42_v1.json",
    ]
    manifests = [(str(path), json.loads(path.read_text())) for path in paths]
    rows, audit = build(train, formal, promotion, manifests, seed=42)
    excluded = set(promotion["real_order"][: promotion["completed_prefix_rows"]])
    for _path, manifest in manifests:
        excluded.update(manifest["indices"])
    index_by_id = {str(row["example_id"]): index for index, row in enumerate(train)}
    selected = {index_by_id[str(row["example_id"])] for row in rows}
    assert not selected.intersection(excluded)
    assert len(rows) == audit["selected_rows"] >= 500
    assert all(row["task"] == "predict_error_persistence" for row in rows)
    assert all(":valid:" in row["example_id"] for row in rows)
    assert audit["candidate_training_overlap"] == 0
    assert audit["formal_example_id_overlap"] == 0
