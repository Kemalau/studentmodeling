from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repair_manifest_builder_is_fail_closed_and_diagnosis_dense():
    source = (ROOT / "scripts/build_state8_stateonly_repair_manifest.py").read_text()
    assert '"eedi_mining": 700' in source
    assert '"malrule": 100' in source
    assert "prior_index_union_overlap" in source
    assert "replay_is_intentional=True" in source
    assert "formal" in source
