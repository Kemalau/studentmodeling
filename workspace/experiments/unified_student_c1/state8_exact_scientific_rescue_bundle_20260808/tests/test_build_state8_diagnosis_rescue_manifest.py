import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_rescue_manifest_is_disjoint_and_well_formed():
    path = ROOT / "manifests/state8_diagnosis_causal_rescue_seed43_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    prior = json.loads(
        (ROOT / "manifests/layerwise_task_retention_eedi_dense_seed42_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["stage"] == "layerwise_task_retention_index_manifest_v1"
    assert value["experiment_stage"] == "state8_diagnosis_causal_rescue_index_manifest_v1"
    assert value["start_global_position"] == 13000
    assert value["segment_rows"] == 1000
    assert value["prior_segment_overlap"] == 0
    assert value["diagnosis_causal_supervision"] is True
    assert len(value["indices"]) == len(set(value["indices"])) == 1000
    assert set(value["indices"]).isdisjoint(prior["indices"])
    assert sum(value["task_counts"].values()) == 1000
    assert value["formal_example_id_overlap"] == 0
